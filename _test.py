class MyClass:
    def my_function(self):
        self.x = 123
        pass

mc = MyClass()
mc.my_function()
if hasattr(mc, 'z'):
    print('MyClass has x')