# -*- coding: utf-8 -*-
"""
Flask-MongoRest operators.

Operators are the building blocks that Resource filters are built upon.
Their role is to generate and apply the right filters to a provided
queryset. For example:

    GET /post/?title__startswith=John

Such request would result in calling this module's `Startswith` operator
like so:

    new_queryset = Startswith().apply(queryset, 'title', 'John')

Where the original queryset would be `BlogPost.objects.all()` and the
new queryset would be equivalent to:

    BlogPost.objects.filter(title__startswith='John')

It's also easy to create your own Operator subclass and use it in your
Resource. For example, if you have an endpoint listing students and you
want to filter them by the range of their scores like so:

    GET /student/?score__range=0,10

Then you can create a Range Operator:

    class Range(Operator):
        op = 'range'
        def prepare_queryset_kwargs(self, field, value, negate=False):
            # For the sake of simplicity, we won't support negate here,
            # i.e. /student/?score__not__range=0,10 won't work.
            lower, upper = value.split(',')
            return {
                field + '__lte': upper,
                field + '__gte': lower
            }

Then you include it in your Resource's filters:

    class StudentResource(Resource):
        document = documents.Student
        filters = {
            'score': [Range]
        }

And this way, the request we mentioned above would result in:

    Student.objects.filter(score__lte=upper, score__gte=lower)
"""

from dateutil.parser import isoparse
from fastnumbers import fast_float

from flask_mongorest.exceptions import ValidationError


def get_bool_value(value, negate):
    if isinstance(value, (bool, int)):
        return not value if negate else value

    lowercase_value = value.lower()
    true = {"true", "1"}
    false = {"false", "0"}

    if lowercase_value in false:
        bool_value = False
    elif lowercase_value in true:
        bool_value = True

    return not bool_value if negate else bool_value


class Operator(object):
    """Base class that all the other operators should inherit from."""

    op = "exact"
    typ = "string"

    # Can be overridden via constructor.
    allow_negation = False

    def __init__(self, allow_negation=False):
        self.allow_negation = allow_negation

    # Lets us specify filters as an instance if we want to override the
    # default arguments (in addition to specifying them as a class).
    def __call__(self):
        return self

    def prepare_queryset_kwargs(self, field, value, negate):
        v = fast_float(value) if self.typ == "number" else value
        parts = [field, "not" if negate else None, self.op]
        return {"__".join(filter(None, parts)): v}

    def apply(self, queryset, field, value, negate=False):
        kwargs = self.prepare_queryset_kwargs(field, value, negate)
        return queryset.filter(**kwargs)


class Ne(Operator):
    op = "ne"


class Lt(Operator):
    op = "lt"
    typ = "number"


class Lte(Operator):
    op = "lte"
    typ = "number"


class Gt(Operator):
    op = "gt"
    typ = "number"


class Gte(Operator):
    op = "gte"
    typ = "number"


class Exact(Operator):
    def prepare_queryset_kwargs(self, field, value, negate):
        # Using <field>__exact causes mongoengine to generate a regular
        # expression query, which we'd like to avoid.
        if negate:
            return {"%s__ne" % field: value}
        else:
            return {field: value}


class IExact(Operator):
    op = "iexact"


class In(Operator):
    allow_negation = True
    op = "in"
    typ = "array"

    def prepare_queryset_kwargs(self, field, value, negate):
        # this is null if the user submits an empty in expression (like
        # "user__in=")
        value = value or []

        # only use 'in' or 'nin' if multiple values are specified
        if "," in value:
            value = value.split(",")
            op = negate and "nin" or self.op
        else:
            op = negate and "ne" or ""
        return {"__".join(filter(None, [field, op])): value}


class Contains(Operator):
    allow_negation = True
    op = "contains"


class IContains(Operator):
    allow_negation = True
    op = "icontains"


class Startswith(Operator):
    allow_negation = True
    op = "startswith"


class IStartswith(Operator):
    allow_negation = True
    op = "istartswith"


class Endswith(Operator):
    allow_negation = True
    op = "endswith"


class IEndswith(Operator):
    allow_negation = True
    op = "iendswith"


class Boolean(Operator):
    typ = "boolean"
    suf = "is"

    def prepare_queryset_kwargs(self, field, value, negate):
        return {field: get_bool_value(value, negate)}


def date_prep(field, value, op):
    try:
        value = isoparse(value)
    except ValueError:
        raise ValidationError("Invalid date format - use ISO 8601")

    return {f"{field}__{op}": value}


class Before(Operator):
    fmt = "date-time"
    suf = "before"
    op = "lt"

    def prepare_queryset_kwargs(self, field, value, negate):
        return date_prep(field, value, self.op)


class After(Operator):
    fmt = "date-time"
    suf = "after"
    op = "gt"

    def prepare_queryset_kwargs(self, field, value, negate):
        return date_prep(field, value, self.op)


class Range(Operator):
    op = "range"

    def prepare_queryset_kwargs(self, field, value, negate=False):
        # NOTE negate not implemented
        lower, upper = value.split(",")
        return {field + "__gte": lower, field + "__lte": upper}


class Size(Operator):
    allow_negation = True
    op = "size"
    typ = "number"


class Exists(Operator):
    op = "exists"
    typ = "boolean"

    def prepare_queryset_kwargs(self, field, value, negate):
        return {f"{field}__{self.op}": get_bool_value(value, negate)}
