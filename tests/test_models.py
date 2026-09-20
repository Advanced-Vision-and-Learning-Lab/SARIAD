"""Model tests that run offline (pretrained weights are never downloaded)."""

import pytest
import torch

from SARIAD.models.components import FeatureGaussianModel
from SARIAD.models.image.MFSA import filters as msfa_filters
from SARIAD.models.image.PadimACE.torch_model import PadimACEModel


# ------------------------------------------------------------------ shared Gaussian base
def test_feature_gaussian_model_fit_and_score():
    from anomalib.models.components import TimmFeatureExtractor

    extractor = TimmFeatureExtractor(backbone="resnet18", layers=["layer1", "layer2"], pre_trained=False)
    model = FeatureGaussianModel(extractor, ["layer1", "layer2"], n_features=40)
    images = torch.rand(6, 3, 64, 64)
    model.train()
    model(images)
    model.fit()
    model.eval()
    out = model(images[:2])
    assert out.anomaly_map.shape == (2, 1, 64, 64) and torch.isfinite(out.anomaly_map).all()
    assert not model.feature_extractor.training                                  # stays frozen in train mode


# ------------------------------------------------------------------ PaDiM-ACE
def _ace_model(cov_type, whitening, n=12, h=4, w=4, n_train=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = h * w
    a = torch.randn(p, n, n, generator=g)
    chol = torch.linalg.cholesky(a @ a.transpose(1, 2) / n + 0.3 * torch.eye(n))
    offset = torch.randn(p, n, generator=g)
    sample = lambda b: (torch.einsum("pij,bpj->bpi", chol, torch.randn(b, p, n, generator=g)) + offset).permute(0, 2, 1).reshape(b, n, h, w)
    model = PadimACEModel(pre_trained=False, n_features=n, cov_type=cov_type, whitening=whitening)
    model.background = sample(n_train)
    model.gaussian.fit(model.background)
    model.signatures = torch.randn(n, p, generator=g) + 2.0
    return model, sample


@pytest.mark.parametrize("cov_type", ["full", "diagonal"])
def test_ace_reference_mode_matches_the_original_implementation(cov_type):
    """Oracle: the original PaDiM-ACE ACELoss computation (SVD of the *inverse* covariance, D^-1/2 U^T)."""
    import torch.nn.functional as F  # noqa: N812

    model, sample = _ace_model(cov_type, "reference")
    test = sample(5)
    ours = model.ace_scores(test).reshape(5, -1)

    mean, inv_cov = model.gaussian.mean, model.gaussian.inv_covariance
    if cov_type == "diagonal":
        inv_cov = torch.diag_embed(torch.diagonal(inv_cov, dim1=-2, dim2=-1))
    u, s, _ = torch.linalg.svd(inv_cov)
    du = torch.diag_embed(s.pow(-0.5)) @ u.transpose(1, 2)
    x_hat = F.normalize(torch.einsum("pij,bpj->bpi", du, (test.reshape(5, test.shape[1], -1) - mean).permute(0, 2, 1)), dim=2)
    s_hat = F.normalize(torch.einsum("pij,jp->pi", du, model.signatures), dim=1)
    reference = torch.einsum("pi,bpi->bp", s_hat, x_hat)
    assert torch.allclose(ours, reference, atol=1e-4)


def test_ace_standard_mode_really_whitens_and_reference_does_not():
    n = 12
    for whitening, whitened in (("standard", True), ("reference", False)):
        model, _ = _ace_model("full", whitening, n=n, n_train=400)
        transform = model._build_transform()
        # the samples the Gaussian was fitted on: their whitened covariance must be ~identity (no sampling noise)
        x = (model.background.reshape(400, n, -1) - model.gaussian.mean).permute(0, 2, 1)
        z = model._whiten(x, transform)
        cov = torch.einsum("bpi,bpj->pij", z, z) / 399
        deviation = (cov - torch.eye(n)).abs().max().item()
        assert (deviation < 0.1) == whitened, (whitening, deviation)


def test_ace_scores_are_cosines_and_peak_on_the_signature():
    model, sample = _ace_model("full", "standard")
    test = sample(6)
    test[:, :, 0, 0] = model.signatures[:, 0]
    scores = model.ace_scores(test)
    assert scores.min() >= -1.0001 and scores.max() <= 1.0001
    assert scores[:, 0, 0, 0].mean() > scores[:, 0].reshape(6, -1)[:, 1:].mean() + 0.3


def test_ace_isotropic_is_plain_cosine():
    model, sample = _ace_model("isotropic", "reference")
    test = sample(4)
    x = torch.nn.functional.normalize((test.reshape(4, test.shape[1], -1) - model.gaussian.mean).permute(0, 2, 1), dim=-1)
    s = torch.nn.functional.normalize(model.signatures.T, dim=-1)
    assert torch.allclose(model.ace_scores(test).reshape(4, -1), torch.einsum("pi,bpi->bp", s, x), atol=1e-5)


def test_ace_arguments_and_errors():
    with pytest.raises(ValueError, match="scoring"):
        PadimACEModel(pre_trained=False, scoring="x")
    model = PadimACEModel(pre_trained=False, backbone="resnet18", layers=["layer1"], n_features=20)
    model.train()
    model(torch.rand(4, 3, 64, 64))
    model.fit()
    model.eval()
    with pytest.raises(RuntimeError, match="signatures"):
        model(torch.rand(1, 3, 64, 64))


def test_ace_checkpoint_roundtrip(tmp_path):
    kwargs = dict(pre_trained=False, backbone="resnet18", layers=["layer1", "layer2"], n_features=30, whitening="standard")
    a = PadimACEModel(**kwargs)
    a.train()
    a(torch.rand(6, 3, 64, 64))
    a.fit()
    a.fit_signatures([torch.rand(4, 3, 64, 64)])
    a.eval()
    torch.save(a.state_dict(), tmp_path / "m.pt")
    b = PadimACEModel(**kwargs)
    b.load_state_dict(torch.load(tmp_path / "m.pt"))
    b.eval()
    x = torch.rand(2, 3, 64, 64)
    assert torch.allclose(a(x).anomaly_map, b(x).anomaly_map, atol=1e-5)


# ------------------------------------------------------------------ MSFA filters / model
def test_msfa_filters_respond_to_edges_and_are_finite():
    image = torch.full((2, 3, 64, 64), 0.2)
    image[:, :, 16:48, 16:48] = 0.8
    gray = image.mean(1, keepdim=True)
    edge = msfa_filters.GradEdge()(gray)
    assert edge[0, 0, 16, 32] > 0.5 and edge[0, 0, 32, 32] < 1e-3
    canny = msfa_filters.Canny()(gray)
    assert canny.shape[1] == 6 and canny[0, 4, 24:40, 24:40].sum() == 0 and canny[0, 4].sum() > 60
    assert msfa_filters.HOG()(gray).shape == (2, 9, 64, 64)
    for names in (("raw",), ("raw", "grad_edge", "hog"), ("raw", "canny", "hog", "grad_edge")):
        fa = msfa_filters.FilterAugmentation(names)
        out = fa(image)
        assert out.shape == (2, fa.out_channels, 64, 64) and torch.isfinite(out).all()
    assert torch.isfinite(msfa_filters.FilterAugmentation(("raw", "grad_edge", "hog", "canny"))(torch.zeros(1, 3, 64, 64))).all()
    with pytest.raises(ValueError):
        msfa_filters.FilterAugmentation(("raw", "wavelet"))


def test_msfa_model_widens_the_first_conv_and_scores():
    from SARIAD.models.image.MFSA.torch_model import MSFAModel

    model = MSFAModel(pre_trained=False, filters=("raw", "grad_edge", "hog"), n_features=30)
    assert model.feature_extractor.backbone.conv1.weight.shape[1] == 11
    images = torch.rand(6, 3, 64, 64)
    model.train()
    model(images)
    model.fit()
    model.eval()
    assert model(images[:2]).anomaly_map.shape == (2, 1, 64, 64)


# ------------------------------------------------------------------ SARATRX anomaly map (needs no submodule)
def test_saratrx_unpatch_inverts_the_submodule_patchify_layout():
    from SARIAD.models.image.SARATRX.anomaly_map import AnomalyMapGenerator

    p, side = 16, 14
    image = torch.rand(2, 1, p * side, p * side)
    # the submodule's MaskedAutoencoder.patchify, verbatim
    h = w = image.shape[2] // p
    x = image.reshape(2, 1, h, p, w, p)
    patches = torch.einsum("nchpwq->nhwpqc", x).reshape(2, h * w, p**2)
    assert torch.equal(AnomalyMapGenerator.unpatch(patches, p, p * side), image)


def test_saratrx_map_averages_over_the_passes_a_patch_was_masked():
    from SARIAD.models.image.SARATRX.anomaly_map import AnomalyMapGenerator

    gen = AnomalyMapGenerator(sigma=1)
    pred, target = torch.zeros(1, 196, 768), torch.ones(1, 196, 768)
    err = gen.patch_error(pred, target)
    assert err.shape == (1, 196, 256) and torch.allclose(err, torch.ones_like(err))
    mask = torch.zeros(1, 196)
    mask[:, :98] = 1
    out = gen(err * mask[..., None] * 2, mask * 2, image_size=224, patch_size=16)   # 2 passes, patches 0..97 masked in both
    assert out.shape == (1, 1, 224, 224) and out[0, 0, :16, :16].mean() > 0.9 and out[0, 0, -16:, -16:].mean() < 0.1
