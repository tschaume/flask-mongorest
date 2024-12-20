# -*- coding: utf-8 -*-
from mongoengine import Document, EmbeddedDocument
from mongoengine.fields import (
    BooleanField,
    DateTimeField,
    EmailField,
    EmbeddedDocumentField,
    IntField,
    ListField,
    ReferenceField,
    StringField,
)


class DateTime(Document):
    datetime = DateTimeField()


class Language(Document):
    name = StringField()


class Person(Document):
    name = StringField()
    languages = ListField(ReferenceField(Language))


class User(Document):
    email = EmailField(unique=True, required=True)
    first_name = StringField(max_length=50)
    last_name = StringField(max_length=50)
    emails = ListField(EmailField())
    datetime = DateTimeField()
    datetime_local = DateTimeField()
    balance = IntField()  # in cents


class Content(EmbeddedDocument):
    text = StringField()
    lang = StringField(max_length=3)


class Post(Document):
    title = StringField(max_length=120, required=True)
    description = StringField(max_length=120, required=False)
    author = ReferenceField(User)
    editor = ReferenceField(User)
    tags = ListField(StringField(max_length=30))
    user_lists = ListField(ReferenceField(User))
    sections = ListField(EmbeddedDocumentField(Content))
    content = EmbeddedDocumentField(Content)
    is_published = BooleanField()

    def primary_user(self):
        return self.user_lists[0] if self.user_lists else None
