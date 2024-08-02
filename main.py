from src import DGM

if __name__ == "__main__":
    model = DGM(yaml="test2.yaml",
                dataset="tadpole", #tadpole, cora...
                full=True)

    model.fit()

