try:
    raise KeyboardInterrupt
finally:# A finally clause is always executed before leaving the try statement,
        # whether an exception has occurred or not
    print('Goodbye, world!')