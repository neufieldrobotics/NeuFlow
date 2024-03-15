import torch


def flow_loss_func(flow_preds, flow_gt, valid,
                   max_flow=400
                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
    valid = (valid >= 0.5) & (mag < max_flow)

    weights = [0.2, 1]

    for i in range(n_predictions):

        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += weights[i] * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    if valid.max() < 0.5:
        pass

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        'mag': mag.mean().item()
    }

    return flow_loss, metrics