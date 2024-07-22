from src import DGM

if __name__ == "__main__":
    model = DGM(yaml="test_e.yaml",
                dataset="tadpole", #tadpole, cora...
                full=False)
