# -*- coding: utf-8 -*-
from setuptools import setup

# Stops exit traceback on tests
# TODO this makes flake8's F401 fail - maybe there's a better way
try:
    import multiprocessing  # noqa
except Exception:
    pass

setup(
    name="flask-mongorest-mpcontribs",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    url="http://github.com/tschaume/flask-mongorest",
    license="BSD",
    author="Close.io",
    author_email="engineering@close.io",
    maintainer="Patrick Huck",
    maintainer_email="phuck@lbl.gov",
    description="Flask restful API framework for MongoDB/MongoEngine",
    long_description="Flask-MongoEngine is a Flask extension that provides integration with MongoEngine, WtfForms and FlaskDebugToolbar.",
    long_description_content_type="text/x-rst",
    packages=["flask_mongorest"],
    package_data={"flask_mongorest": ["templates/mongorest/*"]},
    test_suite="nose.collector",
    zip_safe=False,
    platforms="any",
    install_requires=[
        "boto3",
        "fastnumbers",
        "flask-mongoengine-tschaume>=1.1.0",
        "flask-sse>=1.0.0",
        "flatten-dict",
        "marshmallow-mongoengine>=0.31.0",
        "mimerender-pr36>=0.0.2",
        "pymongo>=3.12.0",
        "python-dateutil",
        "orjson",
    ],
    extras_require={
        'dev': [
            "flake8",
            "pytest",
            "pytest-cov"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
