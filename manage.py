from app import create_app

app = create_app('default')

if __name__ == '__main__':
    # app.run(threaded=True, host='0.0.0.0', port=5001)
    app.run(threaded=True, port=5001)
