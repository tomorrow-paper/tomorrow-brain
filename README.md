# tomorrow-brain

[![](https://api.travis-ci.org/tomorrow-paper/tomorrow-brain.svg?branch=master)](https://travis-ci.org/tomorrow-paper/tomorrow-brain)
[![](http://www.wtfpl.net/wp-content/uploads/2012/12/wtfpl-badge-2.png)](http://www.wtfpl.net/)

`Tomorrow`'s Machine Learning module.

## Quickstart

`tomorrow-brain` relies on `TensorFlow` as a ML backend. You can use the `tomorrow-docker` Docker image to easily setup the required runtime.

```bash
$ docker pull mrkloan/tomorrow
$ docker run --name tomorrow -v $TOMORROW_BRAIN_PATH:/src -td mrkloan/tomorrow bash
$ docker exec -it tomorrow bash

# You're now inside the Docker container

$ cd /src               # Move to the source folder
$ python3 models/*.py   # Regenerate all the models
$ cargo test            # Test the library
```