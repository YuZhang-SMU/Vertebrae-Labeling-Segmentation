class MyConfig(object):

    def __init__(self):
        self.TITLE = ''
        self.DATASET = 'CTSpine1K'
        self.BATCH_SIZE = 2
        self.EPOCH = 200
        self.IR = 0.001
        self.IR_DEAY_EPOCH = 20
        self.IR_DEAY = 0.95

        self.para_q = 32
        self.para_k = 4
        self.para_gamma = 0.95