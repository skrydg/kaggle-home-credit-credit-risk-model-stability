class Dataset:
    def __init__(self, base, depth_0, depth_1, depth_2):
        self.base = base
        self.depth_0 = depth_0
        self.depth_1 = depth_1
        self.depth_2 = depth_2

    def filter(self, filter_lambda):
        self.base = filter_lambda(self.base)

        for i in range(len(self.depth_0)):
            self.depth_0[i] = filter_lambda(self.depth_0[i])
            
        for i in range(len(self.depth_1)):
            self.depth_1[i] = filter_lambda(self.depth_1[i])
        
        for i in range(len(self.depth_2)):
            self.depth_2[i] = filter_lambda(self.depth_2[i])
            
        return self


