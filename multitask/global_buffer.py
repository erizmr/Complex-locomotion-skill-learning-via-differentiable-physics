# import taichi as ti
# from multitask.utils import scalar
#
#
# @ti.data_oriented
# class GlobalBuffer:
#     def __init__(self):
#         self.num_processes = None
#         self.initialized = False
#
#     def initialize(self, config):
#         self.num_processes = config.get_config()["train"]["num_processes"]
#         # Define losses
#         self.loss_batch = scalar()
#         self.loss_velocity_batch = scalar()
#         self.loss_height_batch = scalar()
#         self.loss_pose_batch = scalar()
#         self.loss_rotation_batch = scalar()
#         self.loss_weight_batch = scalar()
#         self.loss_act_batch = scalar()
#         self.loss_dict_batch = {'loss_velocity': self.loss_velocity_batch,
#                                 'loss_height': self.loss_height_batch,
#                                 'loss_pose': self.loss_pose_batch,
#                                 'loss_rotation': self.loss_rotation_batch,
#                                 'loss_weight': self.loss_weight_batch,
#                                 'loss_actuation': self.loss_act_batch}
#         # losses for each batch
#         self.losses_batch = self.loss_dict_batch.values()
#         ti.root.dense(ti.i, self.num_processes).place(self.loss_batch)
#         ti.root.dense(ti.i, self.num_processes).place(*self.losses_batch)
#         self.initialized = True
#         print("Global buffer initialized.")
#
#
# global_buffer = GlobalBuffer()
#
#
