def show_percent(list):
    size = len(list)

    for i, element in enumerate(list):
        print('\u001b[2K\r{:.2f}%'.format((i / size) * 100), end='', flush=True)
        yield element
    
    print('\u001b[2K\r100.00%')
