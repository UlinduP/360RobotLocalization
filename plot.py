import matplotlib.pyplot as plt
import torch

def plot_gt_size_vs_perpendicular_dist(train_df_with_gt):
    plt.scatter(train_df_with_gt['size'], abs(train_df_with_gt['perpendicular_dist']),marker='o',color='b')
    plt.xlabel('Robot Size in the Image (Scaled)')
    plt.ylabel('Perpendicular Distance')
    plt.title('Robot Size vs Perpendicular Distance')
    plt.grid(True)
    plt.show()

def plot_gt_height_vs_perpendicular_dist(train_df_with_gt):
    plt.scatter(train_df_with_gt['height'], abs(train_df_with_gt['perpendicular_dist']),marker='o',color='b')
    plt.xlabel('Robot Height in the Image')
    plt.ylabel('Perpendicular Distance')
    plt.title('Robot Height vs Perpendicular Distance')
    plt.grid(True)
    plt.show()

def plot_gt_width_vs_perpendicular_dist(train_df_with_gt):
    plt.scatter(train_df_with_gt['width'], abs(train_df_with_gt['perpendicular_dist']),marker='o',color='b')
    plt.xlabel('Robot Width in the Image')
    plt.ylabel('Perpendicular Distance')
    plt.title('Robot Width vs Perpendicular Distance')
    plt.grid(True)
    plt.show()

def plot_epochs_vs_loss(epoch_count, train_loss_values, test_loss_values):
    # code to plot train loss and test loss in the same plot
    plt.plot(epoch_count, torch.tensor(train_loss_values).numpy(), label='Train Loss')
    plt.plot(epoch_count, torch.tensor(test_loss_values).numpy(), label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('./plots/1-8-16-32-64-16-8-1-loss_2048X2048imgs_First_Floor.png')
    plt.show()

def plot_image_number_vs_dist(data_df_with_gt):
    # if a robot is present in the image, plot the ground truth distance and if a robot is detected plot both the ground truth and the predicted distance
    for idx, row in data_df_with_gt.iterrows():
        if row['robot_in_the_frame_or_not'] == 1:
            plt.scatter(idx, row['angle'], marker='o', color='g', s=5)
            if row['detected_or_not'] == 1:
                plt.scatter(idx, row['perp_dist_to_the_robot_gt'], marker='o', color='r', s=5)
                plt.scatter(idx, row['perp_dist_to_the_robot_pred'], marker='o', color='b', s=5)
            else:
                plt.scatter(idx, row['perp_dist_to_the_robot_gt'], marker='o', color='r', s=5)

    plt.xlabel('Image Number (Timeline)')
    plt.ylabel('Distance to the Robot (m)')
    plt.title('Image Number vs Distance to the Robot (2024-02-20-16-44-28)')
    plt.grid(True)
    # plt.savefig('./plots/abs_pred_vs_real_2024-02-20-16-44-28_trained_on_2048X2048_first_floor_detection_model_also_trained_with_angle.png')
    plt.show()


