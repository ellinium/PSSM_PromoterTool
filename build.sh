#!/bin/bash
rm dist/*
python3 -m build
twine check dist/*
twine upload dist/*

