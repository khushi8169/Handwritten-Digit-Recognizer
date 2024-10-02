import sys
sys.stdout.reconfigure(encoding='utf-8')
import camera
import train
import predict


def main():
    camera.camera()
    train.train()
    predict.predict()


if __name__=="__main__":
    main()