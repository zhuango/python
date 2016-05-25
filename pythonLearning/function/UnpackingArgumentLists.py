
list1 = list(range(3, 6))
print("list1 = ")
print(list1)

args = [3, 6]
list2 = list(range(*args))
print("list2 = ")
print(list2)

def parrot(voltage, state = 'a stiff', action = 'voom'):
    print("--This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.", end = ' ')
    print("E's", state, "!")
d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
print(parrot(**d))
