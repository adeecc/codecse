.PHONY: all clean

languages := tree-sitter-go tree-sitter-javascript tree-sitter-python tree-sitter-ruby tree-sitter-php tree-sitter-java tree-sitter-c-sharp

all: $(languages)
	~/miniconda3/envs/scs/bin/python build.py

tree-sitter-go:
	git clone --depth 1 https://github.com/tree-sitter/tree-sitter-go

tree-sitter-javascript:
	git clone --depth 1 https://github.com/tree-sitter/tree-sitter-javascript

tree-sitter-python:
	git clone --depth 1 https://github.com/tree-sitter/tree-sitter-python

tree-sitter-ruby:
	git clone --depth 1 https://github.com/tree-sitter/tree-sitter-ruby

tree-sitter-php:
	git clone --depth 1 https://github.com/tree-sitter/tree-sitter-php

tree-sitter-java:
	git clone --depth 1 https://github.com/tree-sitter/tree-sitter-java

tree-sitter-c-sharp:
	git clone --depth 1 https://github.com/tree-sitter/tree-sitter-c-sharp

clear-deps:
	rm -rf tree-sitter-go
	rm -rf tree-sitter-javascript
	rm -rf tree-sitter-python
	rm -rf tree-sitter-ruby
	rm -rf tree-sitter-php
	rm -rf tree-sitter-java
	rm -rf tree-sitter-c-sharp

clean: clear-deps
	rm -rf *.so
