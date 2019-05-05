from collections import OrderedDict

class Module(object):
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def __setattr__(self, name, value):
        if isinstance(value, float):
            # 参数
            self._parameters[name] = value
        elif isinstance(value, Module):
            # 子模块
            self._modules[name] = value
        else:
            # 普通的属性赋值
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # 用于在子类中访问参数或子模块
        if name in self._parameters:
            return self._parameters[name]
        if name in self._modules:
            return self._modules[name]
        
    def register_parameter(self, name, param):
        # 为继承Module的子类显示提供注册参数的函数,和__setattr__功能类似
        self._parameters[name] = param
    
    def named_parameters(self, prefix=""):
        for name, p in self._parameters.items():
            # 保证参数名不重复
            if prefix:
                yield prefix + "." + name, p
            else:
                yield name, p
        
        for mname, module in self._modules.items():
            # 保证参数名不重复
            if prefix:
                submodule_prefix = prefix + "." + mname
            else:
                submodule_prefix = mname
            # 递归调用子模块的named_parameters, 获取子模块的参数
            for name, p in module.named_parameters(submodule_prefix):
                yield name, p
    def parameters(self):
        for name, p in self.named_parameters():
            yield p

class FakeMLP(Module):
    def __init__(self):
        super(FakeMLP, self).__init__()
        # 模型参数
        self.para1 = 0.5
        self.para2 = 0.6


class FakeLSTM(Module):
    def __init__(self):
        super(FakeLSTM, self).__init__()
        # 模型参数
        self.para1 = 0.3
        self.para2 = 0.4
        self.mlp = FakeMLP()

class MyModel(Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 两个自定义参数+三个已有模型
        self.para1 = 0.1
        self.para2 = 0.2
        self.fakeLSTM0 = FakeLSTM()
        self.fakeLSTM1 = FakeLSTM()
        self.fakeLSTM2 = FakeLSTM()
    
    def forward(self):
        # 打印模型所有参数, 包括的子模型参数
        print("##################################################")
        for name, value in self.named_parameters():
            print(name, value)
        print("##################################################")
        
        # 实际使用时,传入优化器的参数
        print(list(self.parameters()))
        # 验证可访问参数和子模块
        print(self.para1)
        print(self.para2)
        print(self.fakeLSTM0)
        print(self.fakeLSTM1)
        print(self.fakeLSTM2)

        # 重新注册名为"fakeLSTM0.mlp.para1"的参数
        self.register_parameter("fakeLSTM0.mlp.para1", 1000000.0)

        print("##################################################")
        for name, value in self.named_parameters():
            print(name, value)
        print("##################################################")
        
# Module   : torch.nn.Module
# FakeLSTM : torch.nn.LSTM
# float    : torch.nn.Parameter
# MyModel  : 自定义的模型
myModel = MyModel()
myModel.forward()
