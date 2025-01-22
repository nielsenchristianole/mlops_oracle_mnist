from oracle_mnist.train import train
import cProfile

if __name__ == "__main__":
    cProfile.run("train()")
    
