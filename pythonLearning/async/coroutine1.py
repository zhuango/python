import types
@types.coroutine
def generator_coroutine():
    yield 1
async def native_coroutine():
    await generator_coroutine()

a = native_coroutine().send(None)
print(a)