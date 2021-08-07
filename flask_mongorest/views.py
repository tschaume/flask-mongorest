# -*- coding: utf-8 -*-
import hashlib
import json
import os
import sys
import time
import traceback
from collections import deque
from gzip import GzipFile
from io import BytesIO, StringIO
from urllib.parse import unquote

import boto3
import mongoengine
from botocore.errorfactory import ClientError
from fdict import fdict
from flask import render_template, request
from flask.views import MethodView
from flask_sse import sse
from mimerender import FlaskMimeRender, register_mime
from werkzeug.exceptions import NotFound, Unauthorized

from flask_mongorest import methods
from flask_mongorest.exceptions import ValidationError
from flask_mongorest.utils import MongoEncoder

BUCKET = os.environ.get("S3_DOWNLOADS_BUCKET", "mongorest-downloads")
CNAME = os.environ.get("PORTAL_CNAME")

s3_client = boto3.client("s3")
flask_mimerender = FlaskMimeRender(global_override_input_key="short_mime")
register_mime("gz", ("application/gzip",))


def render_json(**payload):
    return json.dumps(payload, allow_nan=True, cls=MongoEncoder)


def render_html(**payload):
    d = json.dumps(payload, cls=MongoEncoder, sort_keys=True, indent=4)
    return render_template("mongorest/debug.html", data=d)


def render_gz(**payload):
    s3 = payload.get("s3")

    if s3 and s3["update"]:
        fmt = request.args.get("format")
        if fmt not in {"json", "csv"}:
            raise ValueError(f"Format {fmt} not supported!")

        df_keys = {"data", "columns", "index"}

        for obj in payload["data"]:
            for k in list(obj.keys()):
                if isinstance(obj[k], list):
                    if all("id" in d for d in obj[k]):
                        if fmt == "json":
                            obj[k] = [{"id": d["id"]} for d in obj[k]]
                        else:
                            obj[k] = {str(i): d["id"] for i, d in enumerate(obj[k])}
                    elif fmt == "csv" and k not in df_keys:
                        del obj[k]
                elif isinstance(obj[k], dict) and "id" in obj[k]:
                    obj[k] = {"id": obj[k]["id"]}

        if fmt == "json":
            content_type = "application/json"
            contents = json.dumps(payload["data"], allow_nan=True, cls=MongoEncoder)
        else:
            from pandas import DataFrame, json_normalize

            content_type = "text/csv"
            contents = None

            for obj in payload["data"]:
                if df_keys.issubset(obj.keys()):
                    try:
                        df = DataFrame.from_records(
                            obj["data"], columns=obj["columns"], index=obj["index"]
                        )
                    except Exception as ex:
                        print(str(ex))
                        continue

                    if contents is None:
                        contents = StringIO()
                    else:
                        contents.write("\n\n")

                    meta = json_normalize(
                        {k: v for k, v in obj.items() if k not in df_keys}
                    )
                    meta.to_csv(contents)
                    contents.write("\n")
                    df.to_csv(contents)

            if contents is None:
                contents = json_normalize(payload["data"]).to_csv()

        gzip_buffer = BytesIO()
        if isinstance(contents, StringIO):
            contents = contents.getvalue()

        with GzipFile(mode="wb", fileobj=gzip_buffer) as gzip_file:
            gzip_file.write(
                contents.encode("utf-8")
            )  # need to give full contents to compression

        body = gzip_buffer.getvalue()
        s3_client.put_object(
            Bucket=BUCKET,
            Key=s3["key"],
            ContentType=content_type,
            ContentEncoding="gzip",
            Body=body,
        )
        return body

    retr = s3_client.get_object(Bucket=BUCKET, Key=s3["key"])
    buffer = BytesIO(retr["Body"].read())
    return buffer.getvalue()


try:
    text_type = unicode  # Python 2
except NameError:
    text_type = str  # Python 3


def get_exception_message(e):
    """ME ValidationError has compatibility code with py2.6
    that doesn't follow py3 .args interface. This works around that.
    """
    from mongoengine.errors import ValidationError as MEValidationError

    if isinstance(e, MEValidationError) and not e.args:
        return e.message
    else:
        return e.args[0]


def serialize_mongoengine_validation_error(e):
    """
    Takes a MongoEngine ValidationError as an argument, and returns a
    serializable error dict. Note that we can have nested ValidationErrors.
    """

    def serialize_errors(e):
        if isinstance(e, Exception):
            return get_exception_message(e)
        elif hasattr(e, "items"):
            return {k: serialize_errors(v) for (k, v) in e.items()}
        else:
            return text_type(e)

    if e.errors:
        return {"field-errors": serialize_errors(e.errors)}
    else:
        return {"error": get_exception_message(e)}


class ResourceView(MethodView):
    resource = None
    methods = []
    authentication_methods = []

    def __init__(self):
        assert self.resource and self.methods

    @flask_mimerender(default="json", json=render_json, html=render_html, gz=render_gz)
    def dispatch_request(self, *args, **kwargs):
        # keep all the logic in a helper method (_dispatch_request) so that
        # it's easy for subclasses to override this method (when they don't want to use
        # this mimerender decorator) without them also having to copy/paste all the
        # authentication logic, etc.
        return self._dispatch_request(*args, **kwargs)

    def _dispatch_request(self, *args, **kwargs):
        authorized = True if len(self.authentication_methods) == 0 else False
        for authentication_method in self.authentication_methods:
            if authentication_method().authorized():
                authorized = True
        if not authorized:
            return {"error": "Unauthorized"}, "401 Unauthorized"

        try:
            self._resource = self.requested_resource(request)
            return super(ResourceView, self).dispatch_request(*args, **kwargs)
        except (ValueError, ValidationError, mongoengine.errors.ValidationError) as e:
            return {"error": str(e)}, "400 Bad Request"
        except (Unauthorized, mongoengine.errors.NotUniqueError) as e:
            return {"error": str(e)}, "401 Unauthorized"
        except (NotFound, mongoengine.queryset.DoesNotExist) as e:
            return {"error": str(e)}, "404 Not Found"
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb = traceback.format_exception(exc_type, exc_value, exc_tb)
            err = "".join(tb)
            print(err)
            return {"error": err}, "500 Internal Server Error"

    def handle_validation_error(self, e):
        if isinstance(e, ValidationError):
            raise
        elif isinstance(e, mongoengine.ValidationError):
            raise ValidationError(serialize_mongoengine_validation_error(e))
        else:
            raise

    def requested_resource(self, request):
        """In the case where the Resource that this view is associated with points to a Document class
        that allows inheritance, this method should indicate the specific Resource class to use
        when processing POST and PUT requests through information available in the request
        itself or through other means."""
        # Default behavior is to use the (base) resource class
        return self.resource()

    def get(self, **kwargs):
        pk = kwargs.pop("pk", None)
        short_mime = kwargs.pop("short_mime", None)
        fmt = self._resource.params.get("format")

        # Set the view_method on a resource instance
        if pk:
            self._resource.view_method = methods.Fetch
        elif short_mime:
            if short_mime != "gz":
                raise ValueError(f"{short_mime} not supported")
            self._resource.view_method = methods.Download
        else:
            self._resource.view_method = methods.BulkFetch

        if pk is None:
            extra = {}
            # generate hash/etag and S3 object name

            if self._resource.view_method == methods.Download:
                # Create a queryset filter to control read access to the underlying objects
                doc_fields = self._resource.document._fields.keys()
                only_fields = ["pk"]
                has_last_modified = "last_modified" in doc_fields
                if has_last_modified:
                    only_fields.append("last_modified")

                qfilter = lambda qs: self.has_read_permission(
                    request, qs.only(*only_fields).clone()
                )
                result = self._resource.get_objects(qfilter=qfilter)

                if len(result) == 2:
                    objs, has_more = result
                    extra = {}
                elif len(result) == 3:
                    objs, has_more, extra = result
                else:
                    raise ValueError("Unsupported value of resource.get_objects")

                primary_keys = [str(obj.pk) for obj in objs]
                if has_last_modified:
                    last_modifieds = [
                        obj.last_modified
                        for obj in objs
                        if hasattr(obj, "last_modified")
                    ]
                    last_modified = max(last_modifieds) if last_modifieds else None

                dct = {"ids": primary_keys, "params": self._resource.params}
                sha1 = hashlib.sha1(
                    json.dumps(dct, sort_keys=True).encode("utf-8")
                ).hexdigest()
                filename = f"{sha1}.{fmt}"
                key = f"{CNAME}/{filename}" if CNAME else filename
                extra["s3"] = {"key": key, "update": False}
                try:
                    s3_client_kwargs = dict(Bucket=BUCKET, Key=key)
                    if has_last_modified and last_modified:
                        s3_client_kwargs["IfModifiedSince"] = last_modified
                    s3_client.head_object(**s3_client_kwargs)
                except ClientError:
                    extra["s3"]["update"] = True

                if not extra["s3"]["update"]:
                    return (
                        extra,
                        "200 OK",
                        {
                            "Content-Disposition": f'attachment; filename="{filename}.{short_mime}"'
                        },
                    )

            # Create a queryset filter to control read access to the underlying objects
            qfilter = lambda qs: self.has_read_permission(request, qs.clone())
            result = self._resource.get_objects(qfilter=qfilter)

            # Result usually contains objects and a has_more bool. However, in case where
            # more data is returned, we include it at the top level of the response dict
            if len(result) == 2:
                objs, has_more = result
            elif len(result) == 3:
                objs, has_more, extra_update = result
                extra.update(extra_update)
            else:
                raise ValueError("Unsupported value of resource.get_objects")

            # Serialize the objects one by one
            data = []
            url = unquote(request.url).encode("utf-8")
            channel = hashlib.sha1(url).hexdigest()

            if "s3" not in extra or extra["s3"]["update"]:
                print(f"serializing {channel}...")
                tic = time.perf_counter()
                batch_size, total_count = 1000, extra["total_count"]

                for idx, obj in enumerate(objs):
                    try:
                        data.append(self._resource.serialize(obj, params=request.args))
                    except Exception as e:
                        fixed_obj = self._resource.handle_serialization_error(e, obj)
                        if fixed_obj is not None:
                            data.append(fixed_obj)

                    if idx > 0 and (not idx % batch_size or idx == total_count - 1):
                        toc = time.perf_counter()
                        nobjs = batch_size
                        if idx == total_count - 1:
                            nobjs = total_count - batch_size * int(idx / batch_size)
                        print(
                            f"#{idx} Took {toc - tic:0.4f}s to serialize {nobjs} objects."
                        )
                        if self._resource.view_method == methods.Download:
                            sse.publish(
                                {"message": idx + 1}, type="download", channel=channel
                            )
                        tic = time.perf_counter()

            ret = {"data": data}

            if has_more is not None:
                ret["has_more"] = has_more

            if extra:
                ret.update(extra)
        else:
            # Create a queryset filter to control read access to the underlying objects
            qfilter = lambda qs: self.has_read_permission(request, qs.clone())
            obj = self._resource.get_object(pk, qfilter=qfilter)
            ret = self._resource.serialize(obj, params=request.args)

        if self._resource.view_method == methods.Download:
            sse.publish({"message": 0}, type="download", channel=channel)
            return (
                ret,
                "200 OK",
                {
                    "Content-Disposition": f'attachment; filename="{filename}.{short_mime}"'
                },
            )
        else:
            return ret

    def post(self, **kwargs):
        if kwargs.pop("pk"):
            raise NotFound("Did you mean to use PUT?")

        # Set the view_method on a resource instance
        raw_data = self._resource.raw_data
        if isinstance(raw_data, dict):
            # create single object
            self._resource.view_method = methods.Create
            return self.create_object()
        elif isinstance(raw_data, list):
            limit = self._resource.bulk_update_limit
            if len(raw_data) > limit:
                raise ValidationError(f"Can only create {limit} documents at once")
            raw_data_deque = deque(raw_data)
            self._resource.view_method = methods.BulkCreate
            data = []
            tic = time.perf_counter()
            while len(raw_data_deque):
                self._resource._raw_data = raw_data_deque.popleft()
                data.append(self.create_object(skip_post_save=bool(raw_data_deque)))
                dt = time.perf_counter() - tic
                if dt > 50:
                    break

            count = len(data)
            msg = f"Created {count} objects in {dt:0.1f}s ({count/dt:0.3f}/s)."
            print(msg)
            ret = {"data": data, "count": count}
            if raw_data_deque:
                remain = len(raw_data_deque)
                msg += f" Remaining {remain} objects skipped to avoid Server Timeout."
                ret["warning"] = msg
            return ret, "201 Created"
        else:
            raise ValidationError("wrong payload type")

    def create_object(self, skip_post_save=False):
        self._resource.validate_request()
        try:
            obj = self._resource.create_object(save=False)
        except Exception as e:
            self.handle_validation_error(e)

        # Check if we have permission to create this object
        if not self.has_add_permission(request, obj):
            raise Unauthorized

        self._resource.save_object(
            obj, force_insert=True, skip_post_save=skip_post_save
        )
        ret = self._resource.serialize(obj, params=request.args)
        if self._resource.uri_prefix:
            return ret, "201 Created", {"Location": self._resource._url(str(obj.id))}
        else:
            return ret

    def process_object(self, obj):
        """Validate and update an object"""
        # Check if we have permission to change this object
        if not self.has_change_permission(request, obj):
            raise Unauthorized

        self._resource.validate_request(obj)

        try:
            obj = self._resource.update_object(obj)
        except Exception as e:
            self.handle_validation_error(e)

    def process_objects(self, objs):
        """Update each object in the list, and return the total count of updated objects."""
        tic = time.perf_counter()
        nobjs, count = len(objs), 0
        try:
            for obj in objs:
                self.process_object(obj)
                count += 1
                dt = time.perf_counter() - tic
                if dt > 50:
                    break
        except ValidationError as e:
            e.args[0]["count"] = count
            raise e
        else:
            msg = f"Updated {count} objects in {dt:0.1f}s ({count/dt:0.3f}/s)."
            print(msg)
            ret = {"count": count}
            remain = nobjs - count
            if remain:
                msg += f" Remaining {remain} objects skipped to avoid Server Timeout."
                ret["warning"] = msg
            return ret

    def put(self, **kwargs):
        pk = kwargs.pop("pk", None)

        # Set the view_method on a resource instance
        if pk:
            self._resource.view_method = methods.Update
        else:
            self._resource.view_method = methods.BulkUpdate

        if pk is None:
            # Bulk update where the body contains the new values for certain
            # fields.

            # Currently, fetches all the objects and validates them separately.
            # If one of them fails, a ValidationError for this object will be
            # triggered.
            # Ideally, this would be translated into an update statement for
            # performance reasons and would perform the update either for all
            # objects, or for none, if (generic) validation fails. Since this
            # is a bulk update, only the count of objects which were updated is
            # returned.

            # Get a list of all objects matching the filters, capped at this
            # resource's `bulk_update_limit`
            result = self._resource.get_objects()
            if len(result) == 2:
                objs, has_more = result
            elif len(result) == 3:
                objs, has_more, extra = result

            # Update all the objects and return their count
            ret = self.process_objects(objs)
            ret["has_more"] = has_more
            ret.update(extra)
            return ret
        else:
            obj = self._resource.get_object(pk)
            self.process_object(obj)
            raw_data = fdict(self._resource.raw_data, delimiter=".")
            fields = ",".join(raw_data.keys())
            return self._resource.serialize(obj, params={"_fields": fields})

    def delete_object(self, obj, skip_post_delete=False):
        """Delete an object"""
        # Check if we have permission to delete this object
        if not self.has_delete_permission(request, obj):
            raise Unauthorized

        try:
            self._resource.delete_object(obj, skip_post_delete=skip_post_delete)
        except Exception as e:
            self.handle_validation_error(e)

    def delete_objects(self, objs):
        """Delete each object in the list one by one, and return the total count."""
        tic = time.perf_counter()
        nobjs, count = len(objs), 0
        try:
            # separately delete last object to send skip signal
            for iobj, obj in enumerate(objs):
                skip = iobj < nobjs - 1
                self.delete_object(obj, skip_post_delete=skip)
                count += 1
                dt = time.perf_counter() - tic
                if dt > 50:
                    break
        except ValidationError as e:
            e.args[0]["count"] = count
            raise e
        else:
            msg = f"Deleted {count} objects in {dt:0.1f}s ({count/dt:0.3f}/s)."
            print(msg)
            ret = {"count": count}
            remain = nobjs - count
            if remain:
                msg += f" Remaining {remain} objects skipped to avoid Server Timeout."
                ret["warning"] = msg
            return ret

    def delete(self, **kwargs):
        pk = kwargs.pop("pk", None)

        # Set the view_method on a resource instance
        if pk:
            self._resource.view_method = methods.Delete
        else:
            self._resource.view_method = methods.BulkDelete

        if pk is None:
            result = self._resource.get_objects()
            if len(result) == 2:
                objs, has_more = result
                extra = {}
            elif len(result) == 3:
                objs, has_more, extra = result

            # Delete all the objects and return their count
            ret = self.delete_objects(objs)
            ret["has_more"] = has_more
            ret.update(extra)
            return ret
        else:
            obj = self._resource.get_object(pk)
            self.delete_object(obj)
            return {"count": 1}

    # This takes a QuerySet as an argument and then
    # returns a query set that this request can read
    def has_read_permission(self, request, qs):
        return qs

    def has_add_permission(self, request, obj):
        return True

    def has_change_permission(self, request, obj):
        return True

    def has_delete_permission(self, request, obj):
        return True
