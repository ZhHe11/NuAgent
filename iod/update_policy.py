import torch
import torch.nn as nn
from iod import sac_utils

def optimize_ep(algo, tensors, internal_vars, loss_type='ep_'):         # [loss] 对于q和policy的loss
    _update_loss_qf(algo, tensors, internal_vars, loss_type=loss_type)

    algo._gradient_descent(
        tensors[loss_type + 'LossQf1'] + tensors[loss_type + 'LossQf2'],
        optimizer_keys=[loss_type + 'qf'],
    )
    algo._gradient_descent(
        tensors['forward_loss'],
        optimizer_keys=['predict_encoder'],
    )

    _update_loss_op(algo, tensors, internal_vars, loss_type)
    algo._gradient_descent(
        tensors[loss_type + 'LossSacp'],
        optimizer_keys=[loss_type + 'option_policy'],
    )

    _update_loss_alpha(algo, tensors, internal_vars, loss_type)         # 这个是控制sac的entropy的；
    algo._gradient_descent(
        tensors[loss_type + 'LossAlpha'],
        optimizer_keys=[loss_type + 'log_alpha'],
    )

    sac_utils.update_targets(algo)
    


def _update_loss_op(algo, tensors, v, loss_type):
    zero_option = torch.zeros_like((v['obs'].shape(0), algo.dim_option)).to(algo.device).float()
    processed_cat_obs = algo._get_concat_obs(algo.explore_policy.process_observations(v['obs']), zero_option)
    sac_utils.update_loss_sacp(
        algo, tensors, v,
        obs=processed_cat_obs,
        policy=algo.explore_policy,
        qf1=algo.ep_qf1,
        qf2=algo.ep_qf2,
        alpha=algo.ep_log_alpha,
        target_qf1=algo.ep_target_qf1,
        target_qf2=algo.ep_target_qf2,
        loss_type=loss_type,
    )


def _update_loss_alpha(algo, tensors, v, loss_type):
    sac_utils.update_loss_alpha(
        algo, tensors, v, loss_type=loss_type
    )


    
def _update_loss_qf(algo, tensors, v, loss_type='ep_'):
    # policy_type = "sub_goal_reward"
    # policy_type = "baseline"
    # policy_type = self.method["policy"]
    
    explore_type = "RND"
    
    if explore_type == "RND":
        '''
        zhanghe:
        reward 只有探索reward；
        '''      
        # RND: exploration reward
        predict_next_feature = algo.predict_traj_encoder(v["next_obs"]).mean
        target_next_feature = algo.target_traj_encoder(v["next_obs"]).mean.detach()
                
        exp_reward =  ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).detach()
        forward_mse = nn.MSELoss(reduction='none')
        update_proportion = 0.25
        forward_loss = forward_mse(predict_next_feature, target_next_feature).mean(-1)
        mask = torch.rand(len(forward_loss)).to(algo.device)
        mask = (mask < update_proportion).type(torch.FloatTensor).to(algo.device)
        forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(algo.device))
        
        # goal_reward = ((phi_obs_ - phi_obs) * norm_option).sum(dim=1) + (distance_option - distance_next_option)
        policy_rewards = exp_reward * algo._reward_scale_factor
        
        # update to logs
        tensors.update({
            # 'policy_rewards': policy_rewards.mean(),
            'exp_reward': exp_reward.mean(),
            'forward_loss': forward_loss.mean(),
        })
    zero_option = torch.zeros_like((v['obs'].shape(0), algo.dim_option)).to(algo.device).float()
    processed_cat_obs = algo._get_concat_obs(algo.explore_policy.process_observations(v['obs']), zero_option)
    next_processed_cat_obs = algo._get_concat_obs(algo.explore_policy.process_observations(v['next_obs']), zero_option)

    sac_utils.update_loss_qf(
        algo, tensors, v,
        obs=processed_cat_obs,
        actions=v['actions'],   
        next_obs=next_processed_cat_obs,
        dones=v['dones'],
        rewards=policy_rewards,
        policy=algo.explore_policy,
        qf1=algo.ep_qf1,
        qf2=algo.ep_qf2,
        alpha=algo.ep_log_alpha,
        target_qf1=algo.ep_target_qf1,
        target_qf2=algo.ep_target_qf2,
        loss_type=loss_type,
    )

    v.update({
        loss_type + 'processed_cat_obs': processed_cat_obs,
        loss_type + 'next_processed_cat_obs': next_processed_cat_obs,
    })
    
    