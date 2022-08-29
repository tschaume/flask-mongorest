# -*- coding: utf-8 -*-
import orjson
from collections import defaultdict
from math import isnan
from typing import Pattern

import mongoengine
from bson.dbref import DBRef
from fastnumbers import fast_int
from flask import has_request_context, request, url_for
from flatten_dict import unflatten, flatten
from boltons.iterutils import remap, default_enter
from mongoengine.base.datastructures import BaseDict

try:
    from urllib.parse import urlparse
except ImportError:  # Python 2
    from urlparse import urlparse

try:  # closeio/mongoengine
    from mongoengine.base.proxy import DocumentProxy
    from mongoengine.fields import SafeReferenceField
except ImportError:
    DocumentProxy = None
    SafeReferenceField = None

from mongoengine.fields import (
    DictField,
    EmbeddedDocumentField,
    EmbeddedDocumentListField,
    GenericLazyReferenceField,
    GenericReferenceField,
    LazyReferenceField,
    ListField,
    ReferenceField,
)

try:
    from cleancat import Schema as CleancatSchema
    from cleancat import ValidationError as SchemaValidationError
except ImportError:
    CleancatSchema = None

try:
    from marshmallow.exceptions import ValidationError as MarshmallowValidationError
    from marshmallow.utils import _Missing, get_value, set_value
    from marshmallow_mongoengine import ModelSchema
except ImportError:
    ModelSchema = None
    from glom import glom, assign
    from glom.core import PathAccessError

from flask_mongorest import methods
from flask_mongorest.exceptions import UnknownFieldError, ValidationError
from flask_mongorest.utils import equal, isbound, isint


def enter(path, key, value):
    if isinstance(value, BaseDict):
        return dict(), value.items()

    return default_enter(path, key, value)


def get_with_list_index(o, k):
    try:
        return o[fast_int(k)]
    except ValueError:
        return o[k]


class ResourceMeta(type):
    def __init__(cls, name, bases, classdict):
        if classdict.get("__metaclass__") is not ResourceMeta:
            for document, resource in cls.child_document_resources.items():
                if resource == name:
                    cls.child_document_resources[document] = cls
        type.__init__(cls, name, bases, classdict)


class Resource(object):
    # MongoEngine Document class related to this resource (required)
    document = None

    # List of fields that can (and should by default) be included in the
    # response
    fields = None

    # Dict of original field names (as seen in `fields`) and what they should
    # be renamed to in the API response
    rename_fields = {}

    # CleanCat Schema class (used for validation)
    schema = None

    # List of fields that the objects can be ordered by
    allowed_ordering = []

    # Define whether or not this resource supports pagination
    paginate = True

    # Default limit if no _limit is specified in the request. Only relevant
    # if pagination is enabled.
    default_limit = 100

    # Maximum value of _limit that can be requested (avoids DDoS'ing the API).
    # Only relevant if pagination is enabled.
    max_limit = 100

    # Maximum number of objects which can be bulk-updated by a single request
    bulk_update_limit = 1000  # NOTE also used for bulk delete

    # Map of field names to paginate with according default and maximum limits
    fields_to_paginate = {}

    # Map of field names and Resource classes that should be used to handle
    # these fields (for serialization, saving, etc.).
    related_resources = {}

    # List of field names corresponding to related resources. If a field is
    # mentioned here and in `related_resources`, it can be created/updated
    # from within this resource.
    save_related_fields = []

    # Map of MongoEngine Document classes to Resource class names. Defines
    # which sub-resource should be used for handling a particular subclass of
    # this resource's document.
    child_document_resources = {}

    # Whenever a new document is posted and the system doesn't know the type
    # of it yet, it will choose a default sub-resource for this document type
    default_child_resource_document = None

    # Defines whether MongoEngine's select_related should be used on a
    # filtered query set, pulling all the references efficiently.
    select_related = False

    # allow download formats
    download_formats = []

    # Must start and end with a "/"
    uri_prefix = None

    def __init__(self, view_method=None):
        """
        Initializes a resource. Optionally, a method class can be given to
        view_method (see methods.py) so the resource can behave differently
        depending on the method.
        """
        doc_fields = list(self.document._fields.keys())

        if self.fields is None:
            self.fields = doc_fields
        self._related_resources = self.get_related_resources()
        self._rename_fields = self.get_rename_fields()
        self._reverse_rename_fields = {}
        for k, v in self._rename_fields.items():
            self._reverse_rename_fields[v] = k
        assert len(self._rename_fields) == len(
            self._reverse_rename_fields
        ), "Cannot rename multiple fields to the same name"
        self._normal_filters, self._regex_filters = self.get_filters()
        self._child_document_resources = self.get_child_document_resources()
        self._default_child_resource_document = (
            self.get_default_child_resource_document()
        )
        self.data = None
        self._dirty_fields = None
        self.view_method = view_method
        self._normal_allowed_ordering = [
            o for o in self.allowed_ordering if not isinstance(o, Pattern)
        ]
        self._regex_allowed_ordering = [
            o for o in self.allowed_ordering if isinstance(o, Pattern)
        ]

    @property
    def params(self):
        """
        Return parameters of the request which is currently being processed.
        Params can be passed in two different ways:

        1. As a querystring (e.g. '/resource/?status=active&_limit=10').
        2. As a _params property in the JSON payload. For example:
             { '_params': { 'status': 'active', '_limit': '10' } }
        """
        if not has_request_context():
            # `params` doesn't make sense if we don't have a request
            raise AttributeError

        if not hasattr(self, "_params"):
            if "_params" in self.raw_data:
                self._params = self.raw_data["_params"]
            else:
                try:
                    self._params = request.args.to_dict()
                except AttributeError:  # mocked request with regular dict
                    self._params = request.args
        return self._params

    def _enforce_strict_json(self, val):
        """
        Helper method used to raise a ValueError if NaN, Infinity, or
        -Infinity were posted. By default, json.loads accepts these values,
        but it allows us to perform extra validation via a parse_constant
        kwarg.
        """
        # according to the `json.loads` docs: "parse_constant, if specified,
        # will be called with one of the following strings: '-Infinity',
        # 'Infinity', 'NaN'". Since none of them are valid JSON, we can simply
        # raise an exception here.
        raise ValueError

    @property
    def raw_data(self):
        """Validate and return parsed JSON payload."""
        if not has_request_context():
            # `raw_data` doesn't make sense if we don't have a request
            raise AttributeError

        if not hasattr(self, "_raw_data"):
            if request.method in ("PUT", "POST") or request.data:
                if request.mimetype and "json" not in request.mimetype:
                    raise ValidationError(
                        "Please send valid JSON with a 'Content-Type: application/json' header."
                    )
                if request.headers.get("Transfer-Encoding") == "chunked":
                    raise ValidationError("Chunked Transfer-Encoding is not supported.")

                try:
                    self._raw_data = orjson.loads(request.data.decode("utf-8"))
                    if request.method == "PUT":
                        self._raw_data = unflatten(self._raw_data, splitter="dot")
                except ValueError:
                    raise ValidationError("The request contains invalid JSON.")
                if request.method == "PUT" and not isinstance(self._raw_data, dict):
                    raise ValidationError("JSON data must be a dict.")
            else:
                self._raw_data = {}

        return self._raw_data

    @classmethod
    def uri(self, path):
        """Generate a URI reference for the given path"""
        if self.uri_prefix:
            ret = self.uri_prefix + path
            return ret
        else:
            raise ValueError(
                "Cannot generate URI for resources that do not specify a uri_prefix"
            )

    @classmethod
    def _url(self, path):
        """Generate a complete URL for the given path. Requires application context."""
        if self.uri_prefix:
            url = url_for(self.uri_prefix.lstrip("/").rstrip("/"), _external=True)
            ret = url + path
            return ret
        else:
            raise ValueError(
                "Cannot generate URL for resources that do not specify a uri_prefix"
            )

    def get_fields(self):
        """
        Return a list of fields that should be included in the response
        (unless a `_fields` param didn't include them).
        """
        return self.fields

    @staticmethod
    def get_optional_fields():
        """
        Return a list of fields that can optionally be included in the
        response (but only if a `_fields` param mentioned them explicitly).
        """
        return []

    def get_requested_fields(self, **kwargs):
        """
        Process a list of fields requested by the client and return only the
        ones which are allowed by get_fields and get_optional_fields.

        If `_fields` param is set to '_all', return a list of all the fields
        from get_fields and get_optional_fields combined.
        """
        params = kwargs.get("params", None)

        include_all = False

        # NOTE use list(dict.fromkeys()) below instead of set() to maintain order
        if "fields" in kwargs:
            fields = kwargs["fields"]
            all_fields_set = list(dict.fromkeys(fields))
        else:
            fields = list(self.get_fields())
            all_fields = fields + self.get_optional_fields()
            all_fields_set = list(dict.fromkeys(all_fields))

        if params and "_fields" in params:
            params_fields = params["_fields"].split(",")
            only_fields = list(dict.fromkeys(params_fields))
            if "_all" in only_fields:
                include_all = True
        else:
            only_fields = None

        requested_fields = []
        if include_all or only_fields is None:
            if include_all or self.view_method == methods.Download:
                field_selection = all_fields_set
            else:
                field_selection = fields
            for field in field_selection:
                requested_fields.append(field)
        else:
            for field in only_fields:
                actual_field = self._reverse_rename_fields.get(field, field)
                if actual_field in all_fields_set or any(
                    actual_field.startswith(f) for f in all_fields_set
                ):
                    requested_fields.append(actual_field)

        return requested_fields

    def get_max_limit(self):
        return self.max_limit

    def get_related_resources(self):
        return self.related_resources

    def get_save_related_fields(self):
        return self.save_related_fields

    def get_rename_fields(self):
        """
        @TODO should automatically support model_id for reference fields (only) and model for related_resources
        """
        return self.rename_fields

    def get_child_document_resources(self):
        # By default, don't inherit child_document_resources. This lets us have
        # multiple resources for a child document without having to reset the
        # child_document_resources property in the subclass.
        if "child_document_resources" in self.__class__.__dict__:
            return self.child_document_resources
        else:
            return {}

    def get_default_child_resource_document(self):
        # See comment on get_child_document_resources.
        if "default_child_resource_document" in self.__class__.__dict__:
            return self.default_child_resource_document
        else:
            return None

    def get_filters(self):
        """
        Given the filters declared on this resource, return a mapping
        of all allowed filters along with their individual mappings of
        suffixes and operators.

        For example, if self.filters declares:
            { 'date': [operators.Exact, operators.Gte] }
        then this method will return:
            {
                'date': {
                    '': operators.Exact,
                    'exact': operators.Exact,
                    'gte': operators.Gte
                }
            }
        Then, when a request comes in, Flask-MongoRest will match
        `?date__gte=value` to the 'date' field and the 'gte' suffix: 'gte',
        and hence use the Gte operator to filter the data.
        """
        normal_filters, regex_filters = {}, {}
        for field, operators in getattr(self, "filters", {}).items():
            field_filters = {}

            for op in operators:
                if op.op == "exact":
                    field_filters[""] = op

                fk = op.suf if hasattr(op, "suf") else op.op
                field_filters[fk] = op

            if isinstance(field, Pattern):
                regex_filters[field] = field_filters
            else:
                normal_filters[field] = field_filters
        return normal_filters, regex_filters

    def serialize_field(self, obj, **kwargs):
        if self.uri_prefix and hasattr(obj, "id"):
            return self._url(str(obj.id))
        else:
            return self.serialize(obj, **kwargs)

    def _subresource(self, obj):
        """
        Select and create an appropriate sub-resource class for delegation or
        return None if there isn't one.
        """
        s_class = self._child_document_resources.get(obj.__class__)
        if not s_class and self._default_child_resource_document:
            s_class = self._child_document_resources[
                self._default_child_resource_document
            ]
        if s_class and s_class != self.__class__:
            r = s_class(view_method=self.view_method)
            r.data = self.data
            return r
        else:
            return None

    def get_field_value(self, obj, field_name, field_instance=None, **kwargs):
        """Return a json-serializable field value.

        field_name is the name of the field in `obj` to be serialized.
        field_instance is a MongoEngine field definition.
        **kwargs are just any options to be passed through to child resources serializers.
        """
        has_field_instance = bool(field_instance)

        if not has_field_instance:
            if field_name in self.document._fields:
                field_instance = self.document._fields[field_name]
            elif hasattr(self.document, field_name):
                field_instance = getattr(self.document, field_name)
            else:
                field_instance = None

        # Determine the field value
        if has_field_instance:
            field_value = obj
        elif ModelSchema is None:
            try:
                field_value = getattr(obj, field_name)
            except AttributeError:
                try:
                    field_value = glom(obj, field_name)  # slow
                except PathAccessError:
                    raise UnknownFieldError
        else:
            field_value = get_value(obj, field_name)
            if isinstance(field_value, _Missing):
                raise UnknownFieldError

        return self.serialize_field_value(
            obj, field_name, field_instance, field_value, **kwargs
        )

    def serialize_field_value(
        self, obj, field_name, field_instance, field_value, **kwargs
    ):
        """Select and delegate to an appropriate serializer method based on type of field instance.

        field_value is an actual value to be serialized.
        For other fields, see get_field_value method.
        """
        if isinstance(field_instance, (LazyReferenceField, GenericLazyReferenceField)):
            return field_value and field_value.pk

        if isinstance(
            field_instance,
            (ReferenceField, GenericReferenceField, EmbeddedDocumentField),
        ):
            return self.serialize_document_field(field_name, field_value, **kwargs)

        elif isinstance(field_instance, ListField):
            return self.serialize_list_field(
                field_instance, field_name, field_value, **kwargs
            )

        elif isinstance(field_instance, DictField):
            return self.serialize_dict_field(
                field_instance, field_name, field_value, **kwargs
            )

        elif callable(field_instance):
            return self.serialize_callable_field(
                obj, field_instance, field_name, field_value, **kwargs
            )

        return field_value

    def serialize_callable_field(
        self, obj, field_instance, field_name, field_value, **kwargs
    ):
        """Execute a callable field and return it or serialize
        it based on its related resource defined in the `related_resources` map.
        """
        if isinstance(field_value, list):
            value = field_value
        else:
            if isbound(field_instance):
                value = field_instance()
            elif isbound(field_value):
                value = field_value()
            else:
                value = field_instance(obj)
        if field_name in self._related_resources:
            res = self._related_resources[field_name](view_method=self.view_method)
            if isinstance(value, list):
                return [res.serialize_field(o, **kwargs) for o in value]
            elif value is None:
                return None
            else:
                return res.serialize_field(value, **kwargs)
        return value

    def serialize_dict_field(self, field_instance, field_name, field_value, **kwargs):
        """Serialize each value based on an explicit field type
        (e.g. if the schema defines a DictField(IntField), where all
        the values in the dict should be ints).
        """
        if field_instance.field:
            return {
                key: self.get_field_value(
                    elem, field_name, field_instance=field_instance.field, **kwargs
                )
                for (key, elem) in field_value.items()
            }
        # ... or simply return the dict intact, if the field type
        # wasn't specified
        else:
            return field_value

    def serialize_list_field(self, field_instance, field_name, field_value, **kwargs):
        """Serialize each item in the list separately."""
        if not field_value:
            return []

        field_values = []
        for elem in field_value:
            fv = self.get_field_value(
                elem, field_name, field_instance=field_instance.field, **kwargs
            )
            if fv is not None:
                field_values.append(fv)

        return field_values

    def serialize_document_field(self, field_name, field_value, **kwargs):
        """If this field is a reference or an embedded document, either return
        a DBRef or serialize it using a resource found in `related_resources`.
        """
        if field_name in self._related_resources:
            if field_value:
                res = self._related_resources[field_name](view_method=self.view_method)
                return res.serialize_field(field_value, **kwargs)
        else:
            if DocumentProxy and isinstance(field_value, DocumentProxy):
                # Don't perform a DBRef isinstance check below since
                # it might trigger an extra query.
                return field_value.to_dbref()
            if isinstance(field_value, DBRef):
                return field_value
            return field_value and field_value.to_dbref()

    def serialize(self, obj, **kwargs):
        """
        Given an object, serialize it, turning it into its JSON
        representation.
        """
        if not obj:
            return {}

        # If a subclass of an obj has been called with a base class' resource,
        # use the subclass-specific serialization
        subresource = self._subresource(obj)
        if subresource:
            return subresource.serialize(obj, **kwargs)

        # Get the requested fields
        requested_fields = self.get_requested_fields(**kwargs)

        # Drop the kwargs we don't need any more (we're passing `kwargs` to
        # child resources so we don't want to pass `fields` and `params` that
        # pertain to the parent resource).
        kwargs.pop("fields", None)
        kwargs.pop("params", None)

        # Fill in the `data` dict by serializing each of the requested fields
        # one by one.
        data = {}
        for field in requested_fields:
            # resolve the user-facing name of the field
            renamed_field = self._rename_fields.get(field, field)

            # if the field is callable, execute it with `obj` as the param
            value = None
            if hasattr(self, field) and callable(getattr(self, field)):
                value = getattr(self, field)(obj)

                # if the field is associated with a specific resource (via the
                # `related_resources` map), use that resource to serialize it
                if field in self._related_resources and value is not None:
                    related_resource = self._related_resources[field](
                        view_method=self.view_method
                    )
                    if isinstance(value, mongoengine.document.Document):
                        value = related_resource.serialize_field(value)
                    elif isinstance(value, dict):
                        value = {
                            k: related_resource.serialize_field(v)
                            for (k, v) in value.items()
                        }
                    else:  # assume queryset or list
                        value = [related_resource.serialize_field(o) for o in value]
            else:
                try:
                    value = self.get_field_value(obj, field, **kwargs)
                except UnknownFieldError:
                    try:
                        value = self.value_for_field(obj, field)
                    except UnknownFieldError:
                        pass

            if value is not None:
                if isinstance(value, (float, int)) and isnan(value):
                    value = None

                if ModelSchema is None:
                    assign(data, renamed_field, value, missing=dict)  # slow
                else:
                    set_value(data, renamed_field, value)

        return data

    def handle_serialization_error(self, exc, obj):
        """
        Override this to implement custom behavior whenever serializing an
        object fails.
        """
        pass

    def value_for_field(self, obj, field):
        """
        If we specify a field which doesn't exist on the resource or on the
        object, this method lets us return a custom value.
        """
        raise UnknownFieldError

    def validate_request(self, obj=None):
        """
        Validate the request that's currently being processed and fill in
        the self.data dict that'll later be used to save/update an object.

        `obj` points to the object that's being updated, or is empty if a new
        object is being created.
        """
        # When creating or updating a single object, delegate the validation
        # to a more specific subresource, if it exists
        if (request.method == "PUT" and obj) or request.method == "POST":
            subresource = self._subresource(obj)
            if subresource:
                subresource._raw_data = self._raw_data
                subresource.validate_request(obj=obj)
                self.data = subresource.data
                return

        # Don't work on original raw data, we may reuse the resource for bulk
        # updates.
        self.data = self.raw_data.copy()

        # Do renaming in two passes to prevent potential multiple renames
        # depending on dict traversal order.
        # E.g. if a -> b, b -> c, then a should never be renamed to c.
        fields_to_delete = []
        fields_to_update = {}
        for k, v in self._rename_fields.items():
            if v in self.data:
                fields_to_update[k] = self.data[v]
                fields_to_delete.append(v)
        for k in fields_to_delete:
            del self.data[k]
        for k, v in fields_to_update.items():
            self.data[k] = v

        # If CleanCat schema exists on this resource, use it to perform the
        # validation
        if self.schema:
            if CleancatSchema is None and ModelSchema is None:
                raise ImportError(
                    "Cannot validate schema without CleanCat or Marshmallow!"
                )

            if request.method == "PUT" and obj is not None:
                obj_data = {key: getattr(obj, key) for key in obj._fields.keys()}
            else:
                obj_data = None

            if CleancatSchema is not None:
                try:
                    schema = self.schema(self.data, obj_data)
                    self.data = schema.full_clean()
                except SchemaValidationError:
                    raise ValidationError(
                        {"field-errors": schema.field_errors, "errors": schema.errors}
                    )
            elif ModelSchema is not None:
                try:
                    partial = bool(request.method == "PUT" and obj is not None)
                    self.data = self.schema().load(self.data, partial=partial)
                except MarshmallowValidationError as ex:
                    raise ValidationError(ex.messages)

    def get_queryset(self):
        """
        Return a MongoEngine queryset that will later be used to return
        matching documents.
        """
        document_fields = set(self.fields + self.get_optional_fields())

        if request.method == "PUT":
            # make sure to get full documents for updates
            return self.document.objects.only(*document_fields)
        else:
            requested_fields = self.get_requested_fields(params=self.params)
            requested_root_fields = {f.split(".", 1)[0] for f in requested_fields}
            root_mask = requested_root_fields & document_fields
            mask = []

            for requested_field in requested_fields:
                root_field = requested_field.split(".", 1)[0]
                if root_field in root_mask:
                    if "." in requested_field:
                        field_instance = self.document._fields.get(root_field)
                        if isinstance(field_instance, (ReferenceField, ListField)):
                            raise ValidationError(
                                f"Dot access not supported for {root_field}!"
                            )

                    mask.append(requested_field)

            return self.document.objects.only(*mask)

    def get_object(self, pk, qfilter=None):
        """
        Given a PK and an optional queryset filter function, find a matching
        document in the queryset.
        """
        qs = self.get_queryset()
        # If a queryset filter was provided, pass our current queryset in and
        # get a new one out
        if qfilter:
            qs = qfilter(qs)

        if self.view_method != methods.Download:
            qs = self.apply_field_pagination(qs)

        obj = qs.get(pk=pk)

        # We don't need to fetch related resources for DELETE requests because
        # those requests do not serialize the object (a successful DELETE
        # simply returns a `{}`, at least by default). We still want to fetch
        # related resources for GET and PUT.
        if request.method != "DELETE":
            self.fetch_related_resources(
                [obj], self.get_requested_fields(params=self.params)
            )

        return obj

    def apply_field_pagination(self, qs, params=None):
        """apply field pagination according to `fields_to_paginate`"""
        if params is None:
            params = self.params

        field_attrs = {}
        for field, limits in self.fields_to_paginate.items():
            page = params.get(f"{field}_page", 1)
            per_page = params.get(f"{field}_per_page", limits[0])
            if not isint(page):
                raise ValidationError(f"{field}_page must be an integer.")
            if not isint(per_page):
                raise ValidationError(f"{field}_per_page must be an integer.")

            page, per_page = int(page), int(per_page)
            if per_page > limits[1]:
                raise ValidationError(
                    f"Per-page limit ({per_page}) for {field} too large ({limits[1]})."
                )
            if page < 0:
                raise ValidationError(f"{field}_page must be a non-negative integer.")

            per_page = min(per_page, limits[1])
            start_index = (page - 1) * per_page
            field_attrs[field] = {"$slice": [start_index, per_page]}

        return qs.fields(**field_attrs)

    def fetch_related_resources(self, objs, only_fields=None):
        """
        Given a list of objects and an optional list of the only fields we
        should care about, fetch these objects' related resources.
        """
        # NOTE multiple objects can contain DBRefs to the same related document
        queries = defaultdict(set)
        lookup = defaultdict(lambda: defaultdict(list))

        for obj in objs:
            for field_name in self.related_resources.keys():
                if only_fields and field_name not in only_fields:
                    continue

                field_value = get_value(obj, field_name)

                if isinstance(field_value, DBRef):
                    ref_id = field_value.id
                    queries[field_name].add(ref_id)
                    lookup[field_name][ref_id].append(obj.id)
                elif isinstance(field_value, list):
                    for val in field_value:
                        if isinstance(val, DBRef):
                            queries[field_name].add(val.id)
                            lookup[field_name][val.id].append(obj.id)

        related_fields = list(queries.keys())
        related_objects = {f: defaultdict(list) for f in related_fields}

        for field_name in related_fields:
            doc = self.related_resources[field_name].document
            for d in doc.objects.filter(id__in=list(queries[field_name])):
                obj_ids = lookup[field_name][d.id]
                for obj_id in obj_ids:
                    related_objects[field_name][obj_id].append(d)

        for obj in objs:
            for field_name in related_fields:
                old_value = get_value(obj, field_name)
                rel_objs = related_objects[field_name][obj.id]
                new_value = rel_objs[0] if isinstance(old_value, DBRef) else rel_objs
                set_value(obj, field_name, new_value)

    def apply_filters(self, qs, params=None):
        """
        Given this resource's filters, and the params of the request that's
        currently being processed, apply additional filtering to the queryset
        and return it.
        """
        if params is None:
            params = self.params

        for key, value in params.items():
            # If this is a resource identified by a URI, we need
            # to extract the object id at this point since
            # MongoEngine only understands the object id
            if self.uri_prefix:
                url = urlparse(value)
                uri = url.path
                value = uri.lstrip(self.uri_prefix)

            # special handling of empty / null params
            # http://werkzeug.pocoo.org/docs/0.9/utils/ url_decode returns '' for empty params
            if value == "":
                value = None
            elif value in ['""', "''"]:
                value = ""

            negate = False
            op_name = ""
            parts = key.split("__")
            for i in range(len(parts) + 1, 0, -1):
                field = "__".join(parts[:i])
                try:
                    allowed_operators = self._normal_filters[field]
                except KeyError:
                    for k, v in self._regex_filters.items():
                        m = k.match(field)
                        if m:
                            allowed_operators = v
                            break
                    else:
                        allowed_operators = None
                if allowed_operators:
                    parts = parts[i:]
                    break
            if allowed_operators is None:
                continue

            if parts:
                # either an operator or a query lookup!  See what's allowed.
                op_name = parts[-1]
                if op_name in allowed_operators:
                    # operator; drop it
                    parts.pop()
                else:
                    # assume it's part of a lookup
                    op_name = ""
                if parts and parts[-1] == "not":
                    negate = True
                    parts.pop()

            operator = allowed_operators.get(op_name, None)
            if operator is None:
                continue
            if negate and not operator.allow_negation:
                continue
            if parts:
                field = "{}__{}".format(field, "__".join(parts))
            field = self._reverse_rename_fields.get(field, field)
            qs = operator().apply(qs, field, value, negate)
        return qs

    def apply_ordering(self, qs, params=None):
        """
        Given this resource's allowed_ordering, and the params of the request
        that's currently being processed, apply ordering to the queryset
        and return it.
        """
        if params is None:
            params = self.params

        if self.allowed_ordering:
            obys = params.get("_sort", "").split(",")

            if obys and obys[0]:
                order_bys = []
                kwargs = {}

                for oby in obys:
                    with_sign = oby[0] in {"+", "-"}
                    order_sign = oby[0] if with_sign else "+"
                    order_par = oby[1:] if with_sign else oby

                    if order_par in self._normal_allowed_ordering or any(
                        p.match(order_par) for p in self._regex_allowed_ordering
                    ):
                        order_par = self._reverse_rename_fields.get(order_par, order_par)

                    order_bys.append(f"{order_sign}{order_par}")
                    kwargs[f"{order_par}__exists".replace(".", "__")] = True

                qs = qs.filter(**kwargs).order_by(*order_bys)

        return qs

    def get_skip_and_limit(self, params=None):
        """
        Perform validation and return sanitized values for _skip and _limit
        params of the request that's currently being processed.
        """
        max_limit = self.get_max_limit()
        if params is None:
            params = self.params
        if self.paginate:
            # _limit and _skip validation
            for par in ["_limit", "per_page"]:
                if par in params:
                    if not isint(params[par]):
                        raise ValidationError(
                            f'{par} must be an integer (got "{params[par]}" instead).'
                        )
                    if params[par] and int(params[par]) > max_limit:
                        raise ValidationError(f"Limit {params[par]} too large.")
                    limit = min(int(params[par]), max_limit)
                    break
            else:
                limit = min(int(self.default_limit), max_limit)

            for par in ["_skip", "page"]:
                if par in params:
                    if not isint(params[par]):
                        raise ValidationError(f"{par} must be an integer!")
                    if params[par] and int(params[par]) < 0:
                        raise ValidationError(f"{par} must be a non-negative integer")
                    skip = (
                        int(params[par])
                        if par == "_skip"
                        else (int(params[par]) - 1) * limit
                    )
                    break
            else:
                skip = 0

            # Fetch one more so we know if there are more results.
            return skip, limit
        else:
            return 0, max_limit

    def get_objects(self, qs=None, qfilter=None):
        """
        Return objects fetched from the database based on all the parameters
        of the request that's currently being processed.

        Params:
        - Custom queryset can be passed via `qs`. Otherwise `self.get_queryset`
          is used.
        - Pass `qfilter` function to modify the queryset.
        """
        params = self.params
        extra = {}

        if self.view_method == methods.Download:
            fmt = params.get("format")
            if fmt not in self.download_formats:
                raise ValueError(f"`format` must be one of {self.download_formats}")

        custom_qs = True
        if qs is None:
            custom_qs = False
            qs = self.get_queryset()

        # Apply filters and ordering, based on the params supplied by the request
        qs = self.apply_filters(qs, params)
        qs = self.apply_ordering(qs, params)

        # If a queryset filter was provided, pass our current queryset in and
        # get a new one out
        if qfilter:
            qs = qfilter(qs)

        # set total count
        extra["total_count"] = qs.count()

        # Apply pagination to the queryset (if no custom queryset provided)
        bulk_methods = {methods.BulkUpdate, methods.BulkDelete}
        limit = None
        if self.view_method in bulk_methods:
            # limit the number of objects that can be bulk-updated at a time
            qs = qs.limit(self.bulk_update_limit)
            limit = self.bulk_update_limit
        elif not custom_qs:
            # no need to skip/limit if a custom `qs` was provided
            skip, limit = self.get_skip_and_limit(params)
            qs = qs.skip(skip).limit(limit + 1)  # get one extra to determine has_more
            if self.view_method != methods.Download:
                qs = self.apply_field_pagination(qs, params)
            extra["total_pages"] = int(extra["total_count"] / limit) + bool(
                extra["total_count"] % limit
            )

        # Needs to be at the end as it returns a list, not a queryset
        if self.select_related:
            qs = qs.select_related()

        # Evaluate the queryset
        # cheapest way to convert a queryset to a list: [i for i in qs]
        # list(queryset) uses a count() query to determine length
        # https://github.com/MongoEngine/mongoengine/blob/96802599045432274481b4ed9fcc4fad4ce5f89b/mongoengine/dereference.py#L39-L40
        objs = [i for i in qs]

        # Determine the value of has_more
        has_more = False
        if self.paginate:
            has_more = len(objs) > limit

        if has_more:
            objs = objs[:-1]

        # bulk-fetch related resources for moar speed
        self.fetch_related_resources(objs, self.get_requested_fields(params=params))

        return objs, has_more, extra

    def save_related_objects(self, obj, parent_resources=None, **kwargs):
        if not parent_resources:
            parent_resources = [self]
        else:
            parent_resources += [self]

        if self._dirty_fields:
            for field_name in set(self._dirty_fields) & set(
                self.get_save_related_fields()
            ):
                try:
                    related_resource = self.get_related_resources()[field_name]
                except KeyError:
                    related_resource = None

                field_instance = getattr(self.document, field_name)

                # If it's a ReferenceField, just save it.
                if isinstance(field_instance, ReferenceField):
                    instance = getattr(obj, field_name)
                    if instance:
                        if related_resource:
                            related_resource().save_object(
                                instance, parent_resources=parent_resources
                            )
                        else:
                            instance.save()

                # If it's a ListField(ReferenceField), save all instances.
                if isinstance(field_instance, ListField) and isinstance(
                    field_instance.field, ReferenceField
                ):
                    instance_list = getattr(obj, field_name)
                    for instance in instance_list:
                        if related_resource:
                            related_resource().save_object(
                                instance, parent_resources=parent_resources
                            )
                        else:
                            instance.save()

    def save_object(self, obj, **kwargs):
        signal_kwargs = {
            "skip": kwargs.pop("skip_post_save", False),
            "remaining_time": kwargs.pop("remaining_time", None),
            "dirty_fields": kwargs.pop("dirty_fields", None)
        }
        self.save_related_objects(obj, **kwargs)
        obj.save(signal_kwargs=signal_kwargs, **kwargs).reload()
        self._dirty_fields = None  # No longer dirty.

    def get_object_dict(self, data=None, update=False):
        if data is None:
            data = {}
        data = self.data or data
        filter_fields = set(self.document._fields.keys())
        if update:
            # We want to update only the fields that appear in the request data
            # rather than re-updating all the document's existing/other fields.
            filter_fields &= {
                self._reverse_rename_fields.get(field, field)
                for field in self.raw_data.keys()
            }
        update_dict = {
            field: value for field, value in data.items() if field in filter_fields
        }
        return update_dict

    def create_object(self, data=None, save=True, parent_resources=None):
        update_dict = self.get_object_dict(data)
        obj = self.document(**update_dict)
        self._dirty_fields = update_dict.keys()
        if save:
            self.save_object(obj, force_insert=True)
        return obj

    def update_object(self, obj, data=None, save=True, parent_resources=None):
        subresource = self._subresource(obj)
        if subresource:
            return subresource.update_object(
                obj, data=data, save=save, parent_resources=parent_resources
            )

        update_dict = self.get_object_dict(data, update=True) if save else data

        self._dirty_fields = []

        for field, value in update_dict.items():
            update = False
            field_instance = hasattr(obj, "_fields") and obj._fields.get(field)

            # don't hit DB for comparing ReferenceFields
            if hasattr(obj, "_db_data") and isinstance(field_instance, ReferenceField):
                db_val = obj._db_data.get(field)
                id_from_obj = db_val and getattr(db_val, "id", db_val)
                id_from_data = value and getattr(value, "pk", value)
                if id_from_obj != id_from_data:
                    update = True
            elif hasattr(obj, "_fields"):
                if isinstance(field_instance, DictField):
                    if value is None:
                        update = True
                    else:
                        if obj[field] is None:
                            obj[field] = {}

                        pullout_key = getattr(field_instance, "pullout_key", None)

                        def visit(path, key, val):
                            if isinstance(val, dict) and pullout_key in val:
                                return key, val[pullout_key]

                            return True

                        flat = flatten(remap(obj[field], visit=visit, enter=enter), reducer="dot")

                        for k, v in flatten(value, reducer="dot").items():
                            if k not in flat or str(v) != flat[k]:
                                set_value(obj[field], k, v)
                                self._dirty_fields.append(f"{field}.{k}")

                        continue
                elif (
                    field in self._related_resources
                    and isinstance(field_instance, ListField)
                    and not isinstance(field_instance, EmbeddedDocumentListField)
                ):
                    if value is None:
                        update = True
                    else:
                        if obj[field] is None:
                            obj[field] = []

                        res = self._related_resources[field](
                            view_method=self.view_method
                        )
                        for i, v in enumerate(value):
                            if v is None:
                                continue  # no update requested for list item
                            if i == len(obj[field]):
                                obj[field].append(v)
                            else:
                                res.delete_object(obj[field][i])
                                obj[field][i] = v
                            res.save_object(v)
                elif field_instance.primary_key:
                    raise ValidationError(
                        f"`{field}` is primary key and cannot be updated"
                    )
                elif not equal(getattr(obj, field), value):
                    update = True
            elif not equal(obj.get(field), value):
                update = True

            if update:
                set_value(obj, field, value)
                self._dirty_fields.append(field)

        if save:
            self.save_object(obj, dirty_fields=self._dirty_fields)

        return obj

    def delete_object(self, obj, parent_resources=None, **kwargs):
        signal_kwargs = {
            "skip": kwargs.pop("skip_post_delete", False),
            "remaining_time": kwargs.pop("remaining_time", None)
        }
        obj.delete(signal_kwargs=signal_kwargs)


# Py2/3 compatible way to do metaclasses (or six.add_metaclass)
body = vars(Resource).copy()
body.pop("__dict__", None)
body.pop("__weakref__", None)

Resource = ResourceMeta(Resource.__name__, Resource.__bases__, body)
