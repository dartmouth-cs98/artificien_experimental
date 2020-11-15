from artificienlib.constants import *

#define some standard loss functions

#MSE loss
def mse_with_logits(logits, targets, batch_size):
     """ Calculates mse
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    return (logits - targets).sum() / batch_size

#softmax cross entropy lopp
def softmax_cross_entropy_with_logits(logits, targets, batch_size):
    """ Calculates softmax entropy
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    # numstable logsoftmax
    norm_logits = logits - logits.max()
    log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
    # NLL, reduction = mean
    return -(targets * log_probs).sum() / batch_size

#define standard optimizers

#standard gradient descent
def naive_sgd(param, **kwargs):
    return param - kwargs['lr'] * param.grad

#standard function will set tensors as model parametes
def set_model_params(module, params_list, start_param_idx=0):
    """ Set params list into model recursively
    """
    param_idx = start_param_idx

    for name, param in module._parameters.items():
        module._parameters[name] = params_list[param_idx]
        param_idx += 1

    for name, child in module._modules.items():
        if child is not None:
            param_idx = set_model_params(child, params_list, param_idx)

    return param_idx

#define a standard training plan. Func is your loss function
def training_plan(X, y, batch_size, lr, model_params, func):
    # inject params into model
    set_model_params(model, model_params)

    # forward pass
    logits = model.forward(X)
    
    # loss
    loss = func(logits, y, batch_size)
    
    # backprop
    loss.backward()

    # step
    updated_params = [
        naive_sgd(param, lr=lr)
        for param in model_params
    ]
    
    # accuracy
    pred = th.argmax(logits, dim=1)
    target = th.argmax(y, dim=1)
    acc = pred.eq(target).sum().float() / batch_size

    return (
        loss,
        acc,
        *updated_params
    )

#function to create dummy input parameters to make the trace, build model
def build_model(model):
    model_params = [param.data for param in model.parameters()]  # raw tensors instead of nn.Parameter
    X = th.randn(3, 28 * 28)
    y = nn.functional.one_hot(th.tensor([1, 2, 3]), 10)
    lr = th.tensor([0.01])
    batch_size = th.tensor([3.0])
    _ = training_plan.build(X, y, batch_size, lr, model_params, trace_autograd=True)
    
#define standard averaging plan
def avg_plan(avg, item, num):
    new_avg = []
    for i, param in enumerate(avg):
        new_avg.append((avg[i] * num + item[i]) / (num + 1))
    return new_avg

# Build the Plan
_ = avg_plan.build(model_params, model_params, th.tensor([1.0]))

#function to connect to artificien node
def artificien_connect():
    # PyGrid Node address
    grid = ModelCentricFLClient(id="test", address=gridAddress, secure=False)
    grid.connect() # These name/version you use in worker

#function to send model to node
def send_model(name, version, batch_size, learning_rate, max_updates):

    client_config = {
        "name": name,
        "version": version,
        "batch_size": batch_size,
        "lr": learning_rate,
        "max_updates": max_updates  # custom syft.js option that limits number of training loops per worker
    }

    server_config = {
        "min_workers": 5,

        "max_workers": 5,
        "pool_selection": "random",
        "do_not_reuse_workers_until_cycle": 6,
        "cycle_length": 28800,  # max cycle length in seconds
        "num_cycles": 5,  # max number of cycles
        "max_diffs": 1,  # number of diffs to collect before avg
        "minimum_upload_speed": 0,
        "minimum_download_speed": 0,
        "iterative_plan": True  # tells PyGrid that avg plan is executed per diff
    }

    model_params_state = State(
        state_placeholders=[
            PlaceHolder().instantiate(param)
            for param in model_params
        ]
    )

    response = grid.host_federated_training(
        model=model_params_state,
        client_plans={'training_plan': training_plan},
        client_protocols={},
        server_averaging_plan=avg_plan,
        client_config=client_config,
        server_config=server_config
    )
    
    return print("Host response:", response)

# Helper function to make WS requests
    
def sendWsMessage(data):

    ws = create_connection('ws://' + gridAddress)

    ws.send(json.dumps(data))
    message = ws.recv()
    return json.loads(message)

def check_hosted_model(name, version):
    
    cycle_request = {
        "type": "model-centric/cycle-request",
        "data": {
            "worker_id": auth_response['data']['worker_id'],
            "model": name,
        
            "version": version,
            "ping": 1,
            "download": 10000,
            "upload": 10000,
        }
    }
    cycle_response = sendWsMessage(cycle_request)
    print('Cycle response:', json.dumps(cycle_response, indent=2))

    worker_id = auth_response['data']['worker_id']
    request_key = cycle_response['data']['request_key']
    model_id = cycle_response['data']['model_id'] 
    training_plan_id = cycle_response['data']['plans']['training_plan']
    
    # Model
    req = requests.get(f"http://{gridAddress}/model-centric/get-model?worker_id={worker_id}&request_key={request_key}&model_id={model_id}")
    model_data = req.content
    pb = StatePB()
    pb.ParseFromString(req.content)
    model_params_downloaded = protobuf.serde._unbufferize(hook.local_worker, pb)
    print("Params shapes:", [p.shape for p in model_params_downloaded.tensors()])
    
    # Plan "list of ops"
    req = requests.get(f"http://{gridAddress}/model-centric/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=list")
    pb = PlanPB()
    pb.ParseFromString(req.content)
    plan_ops = protobuf.serde._unbufferize(hook.local_worker, pb)
    print(plan_ops.code)
    print(plan_ops.torchscript)
    
    # Plan "torchscript"
    req = requests.get(f"http://{gridAddress}/model-centric/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=torchscript")
    pb = PlanPB()
    pb.ParseFromString(req.content)
    plan_ts = protobuf.serde._unbufferize(hook.local_worker, pb)
    print(plan_ts.code)
    print(plan_ts.torchscript.code)
    
    # Plan "tfjs"
    req = requests.get(f"http://{gridAddress}/model-centric/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=tfjs")
    pb = PlanPB()
    pb.ParseFromString(req.content)
    plan_tfjs = protobuf.serde._unbufferize(hook.local_worker, pb)
    print(plan_tfjs.code)