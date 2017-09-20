async def read_data(db):
    pass

# It allows interoperability between existing generator-based coroutines in asyncio and native coroutines introduced by this PEP:
# coroutine(fn) 
@types.coruntine
def process_data(db):
    data= yield from read_data(db)

async def read_data(db):
    data = await db.fetch('SELECT ...')
