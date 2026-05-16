class EMA:
    def __init__(self, model, decay=0.9999, burn_in=500):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        self.burn_in = burn_in
        self.step = 0

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()


    def to(self, device):
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)

        for name in self.backup:
            self.backup[name] = self.backup[name].to(device)

        return self

    def update(self):
        self.step += 1

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow

                if self.step < self.burn_in:
                    # During burn-in, just copy the weights directly
                    self.shadow[name] = param.data.clone()
                    continue

                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights to model (for sampling)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights (after sampling)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}