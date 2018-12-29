def early_stop(loss, param):
    print('Last Loss {}; Loss Count:{}'.format(param['last'],param['count']))
    if loss >= param['last']:
        param['count'] += 1
    else:
        param['count'] = 0
    param['last'] = loss
    return param