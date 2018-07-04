## Gunicorn aiohttp PyTorch

A concurrent HTTP server  in order to inference using PyTorch.

If running in GPU server, it will allocate one GPU per worker(process), which is authorized by Gunicorn.

You can assign the number of worker in app.sh file (-w $num_worker)

### run

sh app.sh