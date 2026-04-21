import abc
import torch
import torch.nn.functional as F
from utils.catsample import sample_categorical
from tqdm import tqdm

from utils.utils import get_score_fn

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, labels, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size, save_elements=None):
        sigma, dsigma = self.noise(t)  #total_noise, rate_noise
        score = score_fn(x, sigma, labels)

        # THESE ARE THE KEY ELEMENTS WE WANT TO COLLECT
        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)

        # Save elements if requested (sequence is saved separately in main loop, only score available for Euler)
        if save_elements is not None:
            if 'score' in save_elements:
                save_elements['score'].append(score.clone())

        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size, save_elements=None):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma, labels)

        stag_score = self.graph.staggered_score(score, dsigma)
        # print (stag_score.shape)
        probs = stag_score * self.graph.transp_transition(x, dsigma)

        # Save elements if requested (sequence is saved separately in main loop)
        if save_elements is not None:
            if 'score' in save_elements:
                save_elements['score'].append(score.clone())
            if 'stag_score' in save_elements:
                save_elements['stag_score'].append(stag_score.clone())
            if 'prob' in save_elements:
                save_elements['prob'].append(probs.clone())

        return sample_categorical(probs)


class Denoiser:
    # this is just like the AnalyticPredictor but adapting it
    # to use the last (zero) timestep
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, labels, t, save_elements=None):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma, labels)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]

        # Save elements if requested (sequence is saved separately in main loop)
        if save_elements is not None:
            if 'score' in save_elements:
                save_elements['score'].append(score.clone())
            if 'stag_score' in save_elements:
                save_elements['stag_score'].append(stag_score.clone())
            if 'prob' in save_elements:
                save_elements['prob'].append(probs.clone())

        #return probs.argmax(dim=-1)
        return sample_categorical(probs)

class GibbsCorrector:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise
        self.seed = 1234

    def update_fn_rand(self, score_fn, x, labels, t, csteps=1, thr=0.2):
        sigma, dsigma = self.noise(t)
        B, D = x.size()
        gibbs_gen = torch.Generator(device=x.device).manual_seed(self.seed)
        score = score_fn(x, sigma, labels)

        for _ in range(csteps):

            dd = torch.randint(0, D, (B,),generator=gibbs_gen,device=x.device)
            marg_ratio = score.gather(1,dd.view(B, 1, 1).expand(-1, 1, score.size(-1))).squeeze(1)

            sampling_prob = marg_ratio / torch.sum(marg_ratio, dim=1, keepdim=True)

            # current token value at the chosen dimension: cur[b] = x[b, dd[b]]
            x_cur = x.gather(1, dd.view(B, 1)).squeeze(1)

            # probability assigned to the current token
            p_cur = sampling_prob.gather(1, x_cur.view(B, 1)).squeeze(1)            # (B,)
            # decide whether to resample
            do_resample = p_cur < thr                                             # (B,) bool

            # # Find top-2 probs and indices: (B, 2)
            # top2_p, top2_idx = sampling_prob.topk(2, dim=1)
            # # If current is the argmax, best alternative is 2nd best; else it's best
            # is_top1 = (top2_idx[:, 0] == x_cur)
            # p_best_alt = torch.where(is_top1, top2_p[:, 1], top2_p[:, 0])  # (B,)
            # do_resample = (p_cur - p_best_alt) < thr

            # sample a proposed new token for everyone (cheap), then keep/replace by mask
            proposed = sample_categorical(sampling_prob)                          # (B,)
            new_vals = torch.where(do_resample, proposed, x_cur)                    # (B,)

            # new_vals = sample_categorical(sampling_prob)
            x.scatter_(1, dd.view(B, 1), new_vals.view(B, 1))

        return x

    def update_fn_sys(self, score_fn, x, labels, t, csteps=1, thr=0.2):
        sigma, dsigma = self.noise(t)
        B, D = x.size()
        score = score_fn(x, sigma, labels)

        for _ in range(csteps):
            sampling_prob = score / torch.sum(score, dim=-1, keepdim=True)
            print(sampling_prob.shape)

            # probability assigned to the current token
            p_cur = sampling_prob.gather(2, x.unsqueeze(-1)).squeeze(1)            # (B,D)
            print(p_cur.shape)

            # decide whether to resample
            do_resample = p_cur < thr                                             # (B,D) bool
            print(do_resample.shape)

            proposed = sample_categorical(sampling_prob)                          # (B,D)
            print(proposed.shape)
            
            x = torch.where(do_resample, proposed, x)                             # (B,D)

        return x

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):

    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)

    return sampling_fn


def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x, save_elements_list=None):
    # we have always use euler predictor, but there's also analytic
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model, labels):
        sampling_score_fn = get_score_fn(model, train=False, sampling=True)
        # unless sample_limit() is implemented differently for another graph, this is always random
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        # Initialize element storage if requested
        saved_elements = {}
        if save_elements_list:
            for elem in save_elements_list:
                saved_elements[elem] = []

        # Save initial state
        if saved_elements and 'sequence' in saved_elements:
            saved_elements['sequence'].append(x.clone())

        for i in tqdm(range(steps), total=steps, desc="Diffusion Steps"):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            # Create save_elements dict excluding 'sequence' (we'll save it separately after update)
            predictor_save_elements = {}
            if saved_elements:
                for key in saved_elements:
                    if key != 'sequence':
                        predictor_save_elements[key] = saved_elements[key]

            x = predictor.update_fn(sampling_score_fn, x, labels, t, dt,
                                  save_elements=predictor_save_elements if predictor_save_elements else None)

            # Save sequence state after predictor update
            if saved_elements and 'sequence' in saved_elements:
                saved_elements['sequence'].append(x.clone())

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            # Create save_elements dict excluding 'sequence' (we'll save it separately after update)
            denoiser_save_elements = {}
            if saved_elements:
                for key in saved_elements:
                    if key != 'sequence':
                        denoiser_save_elements[key] = saved_elements[key]

            x = denoiser.update_fn(sampling_score_fn, x, labels, t, save_elements=denoiser_save_elements if denoiser_save_elements else None)

            # Save final sequence state after denoiser
            if saved_elements and 'sequence' in saved_elements:
                saved_elements['sequence'].append(x.clone())

        if saved_elements:
            return x, saved_elements
        return x

    return pc_sampler

def get_gibbs_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, csteps=1, cdiv=1, ctype='sys', thr=0.1, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    corrector = GibbsCorrector(graph, noise)
    denoiser = Denoiser(graph, noise)
    num_steps = steps * cdiv // (cdiv+1)   # in order to match the NFE

    @torch.no_grad()
    def gibbs_sampler(model, labels):
        score_fn = get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps

        for i in tqdm(range(num_steps), total=num_steps, desc="Diffusion Steps"):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = proj_fun(x)
            x = predictor.update_fn(score_fn, x, labels, t, dt)
            if i % cdiv == 0:
                if ctype == 'random':
                    x = corrector.update_fn_rand(score_fn, x, labels, t, csteps=csteps, thr=thr)
                elif ctype == 'sys':
                    x = corrector.update_fn_sys(score_fn, x, labels, t, csteps=csteps, thr=thr)

        if denoise:
            # denoising step
            x = proj_fun(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(score_fn, x, t)

        return x

    return gibbs_sampler
