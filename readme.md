### Run
Please create a venv
```
python -m venv venv
```

Install torch and numpy
```
pip install -r requirements.txt
```

Activate the venv
```
. ./venv/bin/activate
```



Run the program
```
export DYLD_LIBRARY_PATH=$(pwd)/venv/lib/python3.12/site-packages/torch/lib:$DYLD_LIBRARY_PATH
LIBTORCH_USE_PYTORCH=1 cargo run .
```

Build docker image 
```
docker buildx build --platform linux/amd64 -t solar-inference .

```
