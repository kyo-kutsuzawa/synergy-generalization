import numpy as np
from sklearn.decomposition import PCA, NMF


class SpatialSynergy:
    """Spatial synergies.
    """

    def __init__(self, n_synergies, method="nmf"):
        """
        Args:
            n_synergies: Number of synergies
            method: Synergy extraction method PCA or NMF
        """
        self.n_synergies = n_synergies
        self.method = method

        # Initialize variables
        self.model = None
        self.synergies = None
        self.dof = None

    def extract(self, data, max_iter=1000):
        """Extract spatial synergies from given data.

        Data is assumed to have the shape (#trajectories, length, #DoF).
        Synergies have the shape (#synergies, #DoF).
        """
        # Get shape information
        self.dof = data.shape[-1]

        # Reshape given data
        data = data.reshape((-1, self.dof))

        if self.method == "nmf":
            self.model = NMF(n_components=self.n_synergies, max_iter=max_iter)
            self.model.fit(data)
            self.synergies = self.model.components_
        elif self.method == "pca":
            self.model = PCA(n_components=self.n_synergies)
            self.model.fit(data)
            self.synergies = self.model.components_

        return self.synergies

    def encode(self, data):
        """Encode given data to synergy activities.

        Data is assumed to have the shape (#trajectories, length, #DoF).
        Synergy activities have the shape (#trajectories, length, #synergies).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Keep the shape temporarily
        data_shape = data.shape

        # Reshape given data
        data = data.reshape((-1, self.dof))  # shape: (#trajectories * length, #DoF)

        # Encode the data
        activities = self.model.transform(data)

        # Reshape activities
        activities = activities.reshape((data_shape[0], data_shape[1], self.n_synergies))  # shape: (#trajectories, length, #synergies)

        return activities

    def decode(self, activities):
        """Decode given synergy activities to data.

        Synergy activities have the shape (#trajectories, length, #synergies).
        Data is assumed to have the shape (#trajectories, length, #DoF).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Keep the shape temporarily
        act_shape = activities.shape

        # Reshape given activities
        activities = activities.reshape((-1, self.n_synergies))  # shape: (#trajectories * length, #synergies)

        # Decode the synergy activities
        data = self.model.inverse_transform(activities)

        # Reshape reconstruction data
        data = data.reshape((act_shape[0], act_shape[1], self.dof))  # shape: (#trajectories, length, #DoF)

        return data


class SpatioTemporalSynergy:
    """Spatio-temporal synergies.
    """

    def __init__(self, n_synergies, method="nmf"):
        """
        Args:
            n_synergies: Number of synergies
            method: Synergy extraction method pca, nmf, or negative-nmf
        """
        self.n_synergies = n_synergies
        self.method = method

        # Initialize variables
        self.model = None
        self.synergies = None
        self.dof = None
        self.length = None

    def extract(self, data, max_iter=1000):
        """Extract spatio-temporal synergies from given data.

        Data is assumed to have the shape (#data, length, #DoF).
        Synergies have the shape (#synergies, length, #DoF).
        """
        # Get shape information
        self.length = data.shape[1]
        self.dof = data.shape[2]

        # Convert the data to non-negative signals
        if self.method == "negative-nmf":
            data = transform_nonnegative(data)
            self.dof = data.shape[2]  # Update the number of DoF

        # Reshape given data
        data = data.reshape((data.shape[0], -1))  # shape: (#data, length * #DoF)

        if self.method == "nmf" or self.method == "negative-nmf":
            self.model = NMF(n_components=self.n_synergies, max_iter=max_iter)
            self.model.fit(data)
            self.synergies = self.model.components_
            self.synergies = self.synergies.reshape((self.n_synergies, self.length, self.dof))  # Reshape synergies
        elif self.method == "pca":
            self.model = PCA(n_components=self.n_synergies)
            self.model.fit(data)
            self.synergies = self.model.components_
            self.synergies = self.synergies.reshape((self.n_synergies, self.length, self.dof))  # Reshape synergies

        return self.synergies

    def encode(self, data):
        """Encode given data to synergy activities.

        Data is assumed to have the shape (#trajectories, length, #DoF).
        Synergy activities have the shape (#trajectories, #synergies).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Convert the data to non-negative signals
        if self.method == "negative-nmf":
            data = transform_nonnegative(data)

        # Reshape the data from (#trajectories, length, #DoF) to (#trajectories, length * #DoF)
        data = data.reshape((-1, self.length*self.dof))

        # Encode the data
        activities = self.model.transform(data)

        return activities

    def decode(self, activities):
        """Decode given synergy activities to data.

        Synergy activities are assumed to have the shape (#trajectories, #activities).
        Data have the shape (#trajectories, length, #DoF).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Decode the synergy activities
        data = self.model.inverse_transform(activities)
        data = data.reshape((-1, self.length, self.dof))  # Reshape the shape from (#trajectories, length * #DoF) to (#trajectories, length, #DoF)

        # Convert non-negative signals backwards
        if self.method == "negative-nmf":
            data = inverse_transform_nonnegative(data)

        return data


class CombinedSpatioTemporalSynergy:
    """A combination of two spatio-temporal synergy models.

    This class can be used almost the same as `SpatioTemporalSynergy`, while there are some differences;
    this class does not have the `extract` method and `model` member.
    """

    def __init__(self, model1, model2):
        # Original synergy models
        self.model1 = model1
        self.model2 = model2

        # Update synergy-model parameters
        self.n_synergies = model1.n_synergies + model2.n_synergies
        self.synergies = np.concatenate((model1.synergies, model2.synergies), axis=0)
        self.method = model1.method
        self.dof    = model1.dof
        self.length = model1.length

    def encode(self, data):
        """Encode given data to synergy activities.

        Data is assumed to have the shape (#trajectories, length, #DoF).
        Synergy activities have the shape (#trajectories, #synergies).
        """
        # Convert the data to non-negative signals
        if self.method == "negative-nmf":
            data = transform_nonnegative(data)

        # Reshape the data from (#trajectories, length, #DoF) to (#trajectories, length * #DoF)
        data = data.reshape((-1, self.length*self.dof))

        # Encode the data
        activities1 = self.model1.model.transform(data)
        activities2 = self.model2.model.transform(data)

        # Combine synergy activities
        activities = np.concatenate((activities1, activities2), axis=1)
        activities = activities

        return activities

    def decode(self, activities):
        """Decode given synergy activities to data.

        Synergy activities are assumed to have the shape (#trajectories, #activities).
        Data have the shape (#trajectories, length, #DoF).
        """
        # Divide synergy activities
        activities1, activities2 = np.split(activities, [self.model1.n_synergies], axis=1)

        # Decode the synergy activities
        data1 = self.model1.model.inverse_transform(activities1)
        data2 = self.model2.model.inverse_transform(activities2)

        # Combine reconstruction data
        data = data1 + data2
        data = data.reshape((-1, self.length, self.dof))  # Reshape the shape from (#trajectories, length * #DoF) to (#trajectories, length, #DoF)

        # Convert non-negative signals backwards
        if self.method == "negative-nmf":
            data = inverse_transform_nonnegative(data)

        return data


def transform_nonnegative(data):
    """Convert a data that has negative values to non-negative signals with doubled dimensions.

    Data is assumed to have the shape (#trajectories, length, #DoF).
    Converted non-negative data have the shape (#trajectories, length, 2 * #DoF).
    """
    n_dof = data.shape[2]  # Dimensionality of the original data

    # Convert the data to non-negative signals
    data_nn = np.empty((data.shape[0], data.shape[1], n_dof*2))
    data_nn[:, :, :n_dof] = +np.maximum(data, 0.0)
    data_nn[:, :, n_dof:] = -np.minimum(data, 0.0)

    return data_nn


def inverse_transform_nonnegative(data):
    """Inverse conversion of `transform_nonnegative()`; Convert non-negative signals to a data that has negative values.

    Non-negative data is assumed to have the shape (#trajectories, length, 2 * #DoF).
    Reconstructed data have the shape (#trajectories, length, #DoF).
    """
    n_dof = int(data.shape[2] / 2)  # Dimensionality of the original data

    # Restore the original data
    data_rc = np.empty((data.shape[0], data.shape[1], n_dof))
    data_rc = data[:, :, :n_dof] - data[:, :, n_dof:]

    return data_rc


def R2(x, y):
    e = x - y
    v = x - np.mean(x)
    fvu = np.sum(e**2) / np.sum(v**2)
    return 1 - fvu


def _example_spatial():
    """Check synergy extraction code.
    """
    import matplotlib.pyplot as plt

    # Setup constants
    N =  5  # Number of data
    T = 20  # Time length
    M =  6  # Number of DoF
    K =  2  # Number of synergies

    # Create a dataset with shape (N, T, M)
    synergies = np.random.uniform(-1, 1, (M, K))
    activities = np.cumsum(np.random.normal(0, 1.0, (N, T, K)), axis=1)
    data = np.einsum("mk,ntk->ntm", synergies, activities)
    data += np.random.normal(0, 0.1, size=data.shape)  # Add Gaussian noise
    print("Data shape    :", data.shape)

    # Get synergies
    model = SpatialSynergy(K, method="pca")
    model.extract(data)
    print("Synergy shape :", model.synergies.shape)

    # Reconstruct actions
    activities = model.encode(data)
    data_est = model.decode(activities)
    print("Activity shape:", activities.shape)

    # Plot synergy components
    fig = plt.figure()
    fig.suptitle("Synergy components")
    ax1 = fig.add_subplot(1, 1, 1)
    for k in range(K):
        x = np.linspace(-0.5, 0.5, M+2)[1:-1] + k
        ax1.bar(x, model.synergies[k, :], width=0.95/(M+1), linewidth=0, align='center')
    ax1.set_xticks(list(range(K)))
    ax1.set_xticklabels(["synergy #{}".format(k+1) for k in range(K)])
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot reconstruction data
    fig = plt.figure()
    fig.suptitle("Original/Reconstruction data")
    M_row = np.ceil(np.sqrt(M))
    M_col = np.ceil(M/M_row)
    axes = [fig.add_subplot(M_row, M_col, m+1) for m in range(M)]
    for n in range(N):
        for m, ax in enumerate(axes):
            ax.plot(np.arange(data.shape[1]), data[n, :, m], "--", lw=2, color=plt.get_cmap("viridis")((N-n)/(N+1)))
            ax.plot(np.arange(data.shape[1]), data_est[n, :, m],   lw=1, color=plt.get_cmap("viridis")((N-n)/(N+1)))
            ax.set_xlim((0, data.shape[1]-1))
            ax.set_ylabel("DoF #{}".format(m+1))
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


def _example_spatiotemporal():
    """Check synergy extraction code.
    """
    import matplotlib.pyplot as plt

    # Setup constants
    N =  5  # Number of data
    T = 20  # Time length
    M =  3  # Number of DoF
    K =  2  # Number of synergies

    # Create a dataset with shape (N, T, M)
    synergies = np.cumsum(np.random.normal(0, 1, (K, T, M)), axis=1)
    activities = np.random.uniform(-1, 1, (N, K))
    data = np.einsum("ktm,nk->ntm", synergies, activities)
    data += np.random.normal(0, 0.1, size=data.shape)  # Add Gaussian noise
    print("Data shape    :", data.shape)

    # Get synergies
    K *= 2
    model = SpatioTemporalSynergy(K, method="negative-nmf")
    model.extract(data)
    print("Synergy shape :", model.synergies.shape)

    # Reconstruct actions
    activities = model.encode(data)
    data_est = model.decode(activities)
    print("Activity shape:", activities.shape)

    # Plot synergy components
    fig = plt.figure()
    fig.suptitle("Synergy components")
    for k in range(K):
        ax = fig.add_subplot(K, 1, k+1)
        for m in range(M):
            ax.plot(np.arange(model.synergies.shape[1]), model.synergies[k, :, m], color=plt.get_cmap("viridis")((M-m)/(M+1)))
        ax.set_xlim((0, model.synergies.shape[1]-1))
        ax.set_ylabel("synergy #{}".format(k+1))
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot reconstruction data
    fig = plt.figure()
    fig.suptitle("Original/Reconstruction data")
    M_row = np.ceil(np.sqrt(M))
    M_col = np.ceil(M/M_row)
    axes = [fig.add_subplot(M_row, M_col, m+1) for m in range(M)]
    for n in range(N):
        for m, ax in enumerate(axes):
            ax.plot(np.arange(data.shape[1]), data[n, :, m], "--", lw=2, color=plt.get_cmap("viridis")((N-n)/(N+1)))
            ax.plot(np.arange(data.shape[1]), data_est[n, :, m],   lw=1, color=plt.get_cmap("viridis")((N-n)/(N+1)))
            ax.set_xlim((0, data.shape[1]-1))
            ax.set_ylabel("DoF #{}".format(m+1))
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


def _example_combine_synergies():
    import matplotlib.pyplot as plt

    # Setup constants
    N =  5  # Number of data for each synergy model
    T = 20  # Time length
    M =  3  # Number of DoF
    K =  2  # Number of synergies in each synergy model

    # Create a dataset1 with shape (N, T, M)
    synergies1 = np.cumsum(np.random.normal(0, 1, (K, T, M)), axis=1)
    activities1 = np.random.uniform(-1, 1, (N, K))
    data1 = np.einsum("ktm,nk->ntm", synergies1, activities1)
    data1 += np.random.normal(0, 0.1, size=data1.shape)  # Add Gaussian noise

    # Create a dataset2 with shape (N, T, M)
    synergies2 = np.cumsum(np.random.normal(0, 1, (K, T, M)), axis=1)
    activities2 = np.random.uniform(-1, 1, (N, K))
    data2 = np.einsum("ktm,nk->ntm", synergies2, activities2)
    data2 += np.random.normal(0, 0.1, size=data2.shape)  # Add Gaussian noise

    # Get synergies
    K *= 2  # Double the number of synergies
    model1 = SpatioTemporalSynergy(K, method="negative-nmf")
    model2 = SpatioTemporalSynergy(K, method="negative-nmf")
    model1.extract(data1)
    model2.extract(data2)

    # Create a combined synergy model
    model3 = CombinedSpatioTemporalSynergy(model1, model2)

    # Reconstruct actions.
    # Here, some activities are manually set to zero to observe independence of each synergy models.
    # The combined synergy model will succeed in reconstruction if two synergy models are independent to one another.
    activities1 = model3.encode(data1)
    activities2 = model3.encode(data2)
    activities1[:, K:] = 0.0  # Set activities regarding `model2` to zero
    activities2[:, :K] = 0.0  # Set activities regarding `model1` to zero
    data_est1 = model3.decode(activities1)
    data_est2 = model3.decode(activities2)

    # Plot synergy components
    fig = plt.figure()
    fig.suptitle("Synergy components")
    for k in range(K*2):
        ax = fig.add_subplot(K*2, 1, k+1)
        for m in range(M):
            ax.plot(np.arange(model3.synergies.shape[1]), model3.synergies[k, :, m], color=plt.get_cmap("viridis")((M-m)/(M+1)))
        ax.set_xlim((0, model3.synergies.shape[1]-1))
        ax.set_ylabel("synergy #{}".format(k+1))
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot reconstruction data
    fig = plt.figure()
    fig.suptitle("Original/Reconstruction data 1")
    M_row = np.ceil(np.sqrt(M))
    M_col = np.ceil(M/M_row)
    axes = [fig.add_subplot(M_row, M_col, m+1) for m in range(M)]
    for n in range(N):
        for m, ax in enumerate(axes):
            ax.plot(np.arange(data1.shape[1]), data1[n, :, m], "--", lw=2, color=plt.get_cmap("viridis")((N-n)/(N+1)))
            ax.plot(np.arange(data1.shape[1]), data_est1[n, :, m],   lw=1, color=plt.get_cmap("viridis")((N-n)/(N+1)))
            ax.set_xlim((0, data1.shape[1]-1))
            ax.set_ylabel("DoF #{}".format(m+1))
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig = plt.figure()
    fig.suptitle("Original/Reconstruction data 2")
    M_row = np.ceil(np.sqrt(M))
    M_col = np.ceil(M/M_row)
    axes = [fig.add_subplot(M_row, M_col, m+1) for m in range(M)]
    for n in range(N):
        for m, ax in enumerate(axes):
            ax.plot(np.arange(data2.shape[1]), data2[n, :, m], "--", lw=2, color=plt.get_cmap("viridis")((N-n)/(N+1)))
            ax.plot(np.arange(data2.shape[1]), data_est2[n, :, m],   lw=1, color=plt.get_cmap("viridis")((N-n)/(N+1)))
            ax.set_xlim((0, data2.shape[1]-1))
            ax.set_ylabel("DoF #{}".format(m+1))
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


if __name__ == "__main__":
    _example_spatial()
    _example_spatiotemporal()
    _example_combine_synergies()
