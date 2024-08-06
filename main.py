from src import DGM

if __name__ == "__main__":
    model = DGM(yaml="att.yaml",
                dataset="computers", #tadpole, cora...
                full=True,
                device="gpu",
                device_idx=0,
                max_epochs=3
                )

    model.fit()

