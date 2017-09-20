import select
import socket

def coroutine():
    sock = socket.socket()
    sock.setblocking(0)
    address = yield sock
    try:
        sock.connect(address)
    except BlockingIOError:
        pass
    data = yield
    size = yield sock.send(data)
    yield sock.recv(size)

coro = coroutine()
sock = coro.send(None)

wait_list = (sock.fileno(),)

coro.send(('www.baidu.com', 80))
select.select((), wait_list, ())

coro.send(b'Get / HTTP/1.1\r\nHost: www.baidu.com\r\nConnection: Close\r\n\r\n')
select.select(wait_list, (), ())

print(coro.send(1024))
