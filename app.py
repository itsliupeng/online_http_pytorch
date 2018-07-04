# coding=utf-8
import io
import os
import time

import psutil
import torch
from PIL import Image
from aiohttp import web
from resnet_predict import predict as resnet_predict

use_gpu = torch.cuda.is_available()


def get_gpu_id():
    if use_gpu:
        import gpu_stat

        current_pid = os.getpid()
        print('current_pid: ' + str(current_pid))
        sibling_pid_time = sorted(map(lambda x: (x.pid, x.create_time()), psutil.Process(current_pid).parent().children()), key=lambda x: x[1])
        print('sibling_pid_time: ' + str(sibling_pid_time))
        sibling_pids = map(lambda x: x[0], sibling_pid_time)
        for pid in sibling_pids:
            if pid == current_pid:
                return gpu_stat.get_available_gpu_ids()[-1]

            bind_gpu = False
            for gpu_id, pids in gpu_stat.get_pids().items():
                if pid in pids:
                    bind_gpu = True

            if not bind_gpu:
                print('waiting pid %s to bind gpu' % str(pid))
                time.sleep(10)
                return get_gpu_id()
    else:
        return ''


os.environ['CUDA_VISIBLE_DEVICES'] = str(get_gpu_id())
if use_gpu:
    print('get GPU ' + os.environ['CUDA_VISIBLE_DEVICES'])

routes = web.RouteTableDef()


@routes.get('/')
async def index(request):
    return web.json_response('index')


@routes.post('/detect')
async def detect_fn(request):
    data = await request.post()

    if 'file' in data:
        upload_file = data['file']
        # noinspection PyBroadException
        try:
            img = Image.open(io.BytesIO(upload_file.file.read())).convert('RGB')
        except Exception as e:
            return web.json_response({'error_code': -1, 'msg': 'read image error' + str(e)})

        data = resnet_predict(img)
        return web.json_response({'error_code': 0, 'msg': '', 'data': data})

    else:
        return web.json_response({'error_code': -1, 'msg': 'no file found in post request'})


app = web.Application()
app.router.add_routes(routes)
# for gunicorn
application = app

if __name__ == '__main__':
    web.run_app(app, port=8001)
