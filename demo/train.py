from anomalib.engine import Engine
from anomalib.models import Padim

import torch
torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    # load a SAR datamodule
    from SARIAD.datasets import MSTAR
    datamodule = MSTAR()

    # from SARIAD.datasets import HRSID
    # datamodule = HRSID()

    # from SARIAD.datasets import SSDD
    # datamodule = SSDD()

    datamodule.setup()

    i, train_data = next(enumerate(datamodule.train_dataloader()))
    print("Batch Image Shape", train_data.image.shape)

    # load a model
    # model = Padim()

    from SARIAD.models import SARATRX
    model = SARATRX()

    # load a SAR pre processors
    # from SARIAD.pre_processing import SARCNN
    # model = Padim(pre_processor=SARCNN(model=Padim))

    # from SARIAD.pre_processing import NLM
    # model = Padim(pre_processor=NLM(model=Padim))

    # from SARIAD.pre_processing import MedianFilter
    # model = Padim(pre_processor=MedianFilter(model=Padim))

    engine = Engine(max_epochs=30)
    engine.fit(model=model, datamodule=datamodule)

    torch.cuda.empty_cache()

    # predict the whole dataset using latest weights
    predict_results = engine.predict(
        model=model,
        datamodule=datamodule,
    )

    # compute metrics using SARIAD
    from SARIAD.utils import inf
    metrics = inf.Metrics(predict_results)
    print(metrics.get_all_metrics())
    metrics.save_all("results/metrics")

    # run Anomalibs built in test (gets very few metrics)
    # test_results = engine.test(
    #     model=model,
    #     datamodule=datamodule,
    # )
