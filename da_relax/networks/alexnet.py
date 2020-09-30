import sonnet as snt

class AlexNet(snt.AbstractModule):

    def __init__(self):
        super(AlexNet, self).__init__()

    def _build(x, mode='train'):
        if mode == 
