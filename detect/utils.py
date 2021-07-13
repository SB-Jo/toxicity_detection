def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0
    
    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1./norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data**norm_type).sum()
        total_norm = total_norm ** (1./norm_type)
    except Exception as e:
        print(e)

    return total_norm

def read_text(fn):
    with open(fn, 'r', encoding='UTF8') as f:
        lines = f.readlines()
        contexts, responses, targets = [], [], []
        for line in lines[1:]:
            if line.strip() != '':
                target, context, response, _ = line.strip().split('\t')
                targets += [target]
                contexts += [context]
                responses += [response]
    return targets, contexts, responses