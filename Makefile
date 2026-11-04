CUDA    ?= /usr/local/cuda
CC      ?= gcc
CFLAGS  ?= -O2 -g -Wall -Wextra
PREFIX  ?= /usr/local
LDFLAGS_EXTRA ?=

.PHONY: all clean install test
all: build/libgpu_segments.so

build/libgpu_segments.so: src/segment.c
	@mkdir -p build
	$(CC) $(CFLAGS) -fPIC -shared $< -o $@ -I$(CUDA)/include $(LDFLAGS_EXTRA) -lcuda

install: all
	install -D -m 755 build/libgpu_segments.so $(DESTDIR)$(PREFIX)/lib/libgpu_segments.so

test:
	python3 tests/segment_tests.py

clean:
	rm -rf build
