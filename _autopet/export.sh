#!/bin/bash

./build.sh

docker save longilesionlocator | gzip -c > longilesionlocator.tar.gz