
def func():
    def func1():
        print("func1")
    
    def func2():
        print("func2")

    def func3():
        print("func3")

    func3.func1 = func1

    return func3

def funcTest():
    print("funcTest")


f = func()
f.func1()

func.fc = funcTest
func.fc()