from typing import List, Tuple

import torch
from nnunetv2.training.interaction_simulation.agents.all_random import RandomInteractionTypeAgent
from nnunetv2.training.interaction_simulation.agents.base_agent import BaseInteractionAgent
from nnunetv2.training.interaction_simulation.agents.single_interaction_type import SingleInteractionTypeAgent, \
    SingleInteractionAgent
from nnunetv2.training.interaction_simulation.agents.sunk_cost_fallacy import SunkCostInteractionTypeAgent, \
    SunkCostInteractionAgent
from nnunetv2.training.interaction_simulation.interaction_channel_config import InteractionChannelConfig
from nnunetv2.training.interaction_simulation.interactions.base_interaction import BaseInteraction
from nnunetv2.training.interaction_simulation.interactions.delete_non_target_objects import \
    DeleteNonTargetObjectsInteraction
from nnunetv2.training.interaction_simulation.interactions.initial_seg_correction import InitialSegCorrectionInteraction
from nnunetv2.training.interaction_simulation.interactions.lasso import LassoInteraction
from nnunetv2.training.interaction_simulation.interactions.scribblev3 import FabiansScribblev2
from nnunetv2.training.nnUNetTrainer.nnInteractive.interaction_combinations.v2_final import \
    nnInteractiveTrainerV2_educatedGuess2_pseudolabels_CVPRAugs_lr1en3_predCorrupt_bbox2d_lasso_scr_pts_ptsInScr


class nnInteractiveTrainerV2_educatedGuess2_pseudolabels_CVPRAugs_lr1en3_predCorrupt_lassso_scribble(
    nnInteractiveTrainerV2_educatedGuess2_pseudolabels_CVPRAugs_lr1en3_predCorrupt_bbox2d_lasso_scr_pts_ptsInScr):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.interaction_channel_config = InteractionChannelConfig(
            channel_mapping={'lasso': (1, 2), 'scribble': (3, 4)}
        )

    def define_interactions_and_agents(self) -> Tuple[
        List[BaseInteraction], List[float], List[BaseInteractionAgent], List[float]]:
        # we don't break up error masks in followup interactions too often because it's actually realistic to draw a
        # lasso around the thing and breaking up things can break things
        lasso2d = LassoInteraction(interaction_decay=1, p_pick_random_component=0.9,
                                   p_break_up_followup_interactions=0.3, break_up_perlin_smooth=(16, 32),
                                   p_dilate=0.85)

        scribbles = FabiansScribblev2(
            p_center_scribble=2 / 9,  # center is a bit too boring so we reduce its probability
            p_line_scribble=4 / 9,
            p_border_scribble=1/3,
            interaction_decay=1,
            center_affinity=1.5,
            p_pick_random_component=1,
            mask_smooth_sigma=(0.1, 0.2),
            line_thickness=2,
            contour_erosion_radius=(0.00, 0.15),
            p_dilate_fp=0.2,
            # we leave this enabled because it mimics users FP correction (broad strike that goes beyond FP into TN areas)
            fp_dilation_radius=(1, 21),
            fp_dilation_step_radius=3,
            remove_outside_voxels=False,  # this needs to be False to allow outside scribbles
            percent_of_scribble_outside_allowed=0.12  # 12% can be outside
        )

        # initial_seg_slice = InitialSegSliceInteraction()
        # initial_seg_slice_orth = InitialSegSliceInteraction(make_three_orthogonal_slices=True)

        list_of_interactions_for_agent = [
                lasso2d,
                scribbles
            ]

        delete_non_target_objects = DeleteNonTargetObjectsInteraction(
            list_of_interactions_for_agent,
            [
                1 / 2,  # lasso
                1 / 2   # scribbles
            ]
        )
        initial_seg_with_correction = InitialSegCorrectionInteraction(
            list_of_interactions_for_agent,
            [
                 1 / 2, # lasso
                 1 / 2  # scribbles
             ],
            p_correction=0.5
        )

        initial_seg_blocky_with_correction = InitialSegCorrectionInteraction(
            list_of_interactions_for_agent,
            [
                 1 / 2, # lasso
                 1 / 2  # scribbles
             ],
            p_correction=1,
            p_make_blocky=1
        )
        initial_seg_blocky_without_correction = InitialSegCorrectionInteraction(
            list_of_interactions_for_agent,
            [
                 1 / 2, # lasso
                 1 / 2  # scribbles
             ],
            p_correction=0.0,
            p_make_blocky=1
        )

        list_of_initial_interactions = [
            lasso2d,
            scribbles,
            initial_seg_with_correction,
            initial_seg_blocky_with_correction,
            initial_seg_blocky_without_correction,
            delete_non_target_objects
        ]

        initial_interaction_probabilities = [
            0.425,     # lasso
            0.425,       # scribbles
            0.125 / 2, # initial_seg_with_correction
            0.05 / 2, # initial_seg_blocky_with_correction
            0.05 / 2, # initial_seg_blocky_without_correction
            0.075 / 2, # delete_non_target_objects
        ]

        agent_list = [
            # pick a random interaction type and selects a random interaction from that type
            RandomInteractionTypeAgent(list_of_interactions_for_agent, interaction_channel_config=self.interaction_channel_config, interaction_decay=self.interaction_decay),
            # pick a single interaction type and keep using it (random interaction of that type in each iteration)
            SingleInteractionTypeAgent(list_of_interactions_for_agent,
                                       interaction_channel_config=self.interaction_channel_config, interaction_decay=self.interaction_decay),
            # picks whatever the last interaction was and keeps reusing this, never changing
            SingleInteractionAgent(list_of_interactions_for_agent, interaction_channel_config=self.interaction_channel_config, interaction_decay=self.interaction_decay),
            # keep reusing the last interaction type (random interaction of that type picked in each iter), switch to a new type with low prob
            SunkCostInteractionTypeAgent(list_of_interactions_for_agent,
                                         interaction_channel_config=self.interaction_channel_config, interaction_decay=self.interaction_decay, switcheroo_p=0.3),
            SunkCostInteractionTypeAgent(list_of_interactions_for_agent,
                                         interaction_channel_config=self.interaction_channel_config, interaction_decay=self.interaction_decay, switcheroo_p=0.1),
            # uses the same interaction class until switched
            SunkCostInteractionAgent(list_of_interactions_for_agent, interaction_channel_config=self.interaction_channel_config, interaction_decay=self.interaction_decay, switcheroo_p=0.2)
        ]
        agent_probabilities = [
            0.1,
            0.3,
            0.1,
            0.2,
            0.2,
            0.1
        ]

        return list_of_initial_interactions, initial_interaction_probabilities, agent_list, agent_probabilities
