import torch

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(t.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, ref, illu, color, seq, c_model, b, **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1]

        et = c_model(ref, xt, illu, color, t.float())
        x0_t = (xt - et * (1 - at).sqrt()) / (at.sqrt())
        x0_preds.append(x0_t)

        xt_next = at_next.sqrt() * x0_t + ((1 - at_next).sqrt()) * et

        xs.append(xt_next)

    return xs, x0_preds

