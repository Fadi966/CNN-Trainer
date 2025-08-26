from cnn_trainer_app import CNNTrainerApp

def main():
    return CNNTrainerApp('CNN Trainer', 'org.example.cnntrainer')

if __name__ == '__main__':
    app = main()
    app.main_loop()