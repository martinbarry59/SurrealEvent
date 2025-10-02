from dataclasses import dataclass
import math
import torch
@dataclass
class PersistentNoiseConfig:
    hot_pixels: int = 500
    hot_pixel_rate: int = 6          # events per time step per hot pixel
    flicker_sources: int = 4
    flicker_freq_hz: float = 60.0
    flicker_events: int = 300
    # Lamp-like blob settings
    lamp_blobs: int = 3              # Number of lamp-like light sources
    lamp_radius: int = 15            # Radius of lamp illumination
    lamp_events: int = 800           # Higher event count for brighter lamps
    lamp_drift_speed: float = 0.5    # Very slow drift (like swaying lamp posts)
    lamp_intensity_var: float = 0.3  # Intensity variation over time
    # Moving blob settings (for other moving light sources)
    moving_blobs: int = 1
    blob_radius: int = 8
    blob_events: int = 300
    blob_speed: float = 4.0          # Reduced speed for more realistic movement
    drifting_edges: int = 1
    edge_events: int = 100           # Increased for more visibility
    grid_enabled: bool = True
    grid_spacing: int = 32
    grid_events: int = 50            # Increased for more realistic interference
    polarity_consistent: bool = False # Mixed polarity for more realistic noise
    max_total_events: int = 3000     # Increased to accommodate more events
    prob_enable: float = 0.8         # Slightly reduce probability for variation
    
    # Persistent augmentation settings
    enable_augmentations: bool = True
    aug_prob: float = 0.5            # Probability of applying augmentations
    temporal_jitter_prob: float = 0.4
    temporal_jitter_std: float = 0.1  # 10% temporal noise std
    spatial_jitter_prob: float = 0.3
    spatial_jitter_std: float = 10.0  # Pixel noise std
    dropout_prob: float = 0.4
    dropout_rate_range: tuple = (0.1, 0.3)  # 10-30% dropout
    cluster_dropout_prob: float = 0.5  # Probability of cluster vs random dropout
    polarity_flip_prob: float = 0.2
    polarity_flip_rate_range: tuple = (0.1, 0.3)  # 10-30% polarity flips
    rate_variation_prob: float = 0.25
    rate_factor_range: tuple = (0.5, 1.5)  # 0.5-1.5x rate variation

class PersistentNoiseGenerator:
    def __init__(self, width, height, device, config=None, seed=None, training=True):
        self.width = width
        self.height = height
        self.device = device
        self.training = training
        self.cfg = config or PersistentNoiseConfig()
        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(seed)
        self.active = False
        # Persistent augmentation state
        self.augmentation_state = {}
        self.reset()

    def reset(self):
        with torch.no_grad():
            # Decide if noise is enabled this sequence
            self.active = torch.rand(1, generator=self.rng, device=self.device).item() < self.cfg.prob_enable
            
            # Initialize persistent augmentation state
            self._init_persistent_augmentations()
            
            if not self.active:
                return
            c = self.cfg
            # Hot pixels - clustered in realistic patterns
            self.hot_xy = torch.stack([
                torch.randint(0, self.width, (c.hot_pixels,), generator=self.rng, device=self.device),
                torch.randint(0, self.height, (c.hot_pixels,), generator=self.rng, device=self.device)
            ], dim=1)
            # Add some spatial clustering for hot pixels (more realistic)
            if c.hot_pixels > 20:
                cluster_centers = torch.randint(0, min(self.width, self.height)//4, (c.hot_pixels//10, 2), 
                                              generator=self.rng, device=self.device) * 4
                cluster_size = 8
                for i, center in enumerate(cluster_centers):
                    start_idx = i * 10
                    end_idx = min((i + 1) * 10, c.hot_pixels)
                    cluster_pixels = start_idx + torch.arange(end_idx - start_idx, device=self.device)
                    offsets = torch.randint(-cluster_size, cluster_size+1, (len(cluster_pixels), 2), 
                                          generator=self.rng, device=self.device)
                    self.hot_xy[cluster_pixels] = (center.unsqueeze(0) + offsets).clamp(
                        torch.tensor([0, 0], device=self.device), 
                        torch.tensor([self.width-1, self.height-1], device=self.device)
                    )
            
            self.hot_pol = self._pol(c.hot_pixels) if c.polarity_consistent else self._pol(c.hot_pixels, random=True)
            
            # Flicker sources (centers + phase) - positioned like street lights
            self.flicker_centers = torch.stack([
                torch.randint(self.width//8, 7*self.width//8, (c.flicker_sources,), generator=self.rng, device=self.device),
                torch.randint(self.height//8, 3*self.height//8, (c.flicker_sources,), generator=self.rng, device=self.device)  # Upper portion
            ], dim=1)
            self.flicker_phase = torch.rand(c.flicker_sources, device=self.device, generator=self.rng) * 2*math.pi
            
            # Lamp-like blobs (stationary or very slow moving)
            self.lamp_pos = torch.stack([
                torch.rand(c.lamp_blobs, device=self.device, generator=self.rng) * self.width,
                torch.rand(c.lamp_blobs, device=self.device, generator=self.rng) * self.height * 0.6  # Upper 60% of image
            ], dim=1)
            # Very slow random drift for lamps (like swaying in wind)
            theta = torch.rand(c.lamp_blobs, device=self.device, generator=self.rng) * 2 * math.pi
            self.lamp_vel = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) * c.lamp_drift_speed
            # Each lamp has its own intensity variation phase
            self.lamp_intensity_phase = torch.rand(c.lamp_blobs, device=self.device, generator=self.rng) * 2 * math.pi
            
            # Moving blobs (vehicles, etc.)
            self.blob_pos = torch.stack([
                torch.rand(c.moving_blobs, device=self.device, generator=self.rng)*self.width,
                torch.rand(c.moving_blobs, device=self.device, generator=self.rng)*self.height
            ], dim=1)
            theta = torch.rand(c.moving_blobs, device=self.device, generator=self.rng)*2*math.pi
            self.blob_vel = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) * c.blob_speed
            
            # Drifting edges
            self.edge_orient = torch.randint(0,2,(c.drifting_edges,), generator=self.rng, device=self.device)  # 0=H,1=V
            self.edge_pos0 = torch.rand(c.drifting_edges, device=self.device, generator=self.rng)
            self.edge_dir = torch.where(torch.rand(c.drifting_edges, device=self.device, generator=self.rng)>0.5, 1.0, -1.0)
            # Grid
            self.grid_x = torch.arange(0, self.width, c.grid_spacing, device=self.device) if c.grid_enabled else None
            self.grid_y = torch.arange(0, self.height, c.grid_spacing, device=self.device) if c.grid_enabled else None

    def _pol(self, n, random=False):
        if random:
            return (torch.randint(0,2,(n,), generator=self.rng, device=self.device)*2 -1).float()
        return torch.ones(n, device=self.device)

    def step(self, batch_size, t_min, t_max):
        """
        batch_size: number of batches to generate noise for
        t_min: minimum normalized time in [0,1] for this slice
        t_max: maximum normalized time in [0,1] for this slice
        Returns tensor [B, M, 4] (t,x,y,p) appended noise events (t uniformly distributed)
        """

        with torch.no_grad():
            if not self.active:
                return None
            c = self.cfg
            device = self.device
            events = []
            
            # Hot pixels (more realistic with temporal bursting)
            if c.hot_pixels>0:
                # Create bursts of activity for hot pixels (more realistic)
                time_mid = torch.Tensor([(t_min + t_max) / 2]).to(device)
                burst_probability = 0.3 + 0.4 * torch.sin(time_mid * 10.0)  # Varying burst activity
                
                if torch.rand(1, device=device, generator=self.rng).item() < burst_probability:
                    # Select subset of hot pixels to be active this time slice
                    active_fraction = 0.4 + 0.4 * torch.rand(1, device=device, generator=self.rng).item()
                    n_active = int(c.hot_pixels * active_fraction)
                    active_indices = torch.randperm(c.hot_pixels, generator=self.rng, device=device)[:n_active]
                    active_hot_pixels = self.hot_xy[active_indices]
                    
                    # Variable event rate per hot pixel
                    events_per_pixel = torch.randint(1, c.hot_pixel_rate * 2, (n_active,), 
                                                   generator=self.rng, device=device)
                    total_events = events_per_pixel.sum().item()
                    
                    # Generate times with slight clustering (burst behavior)
                    base_times = t_min + (t_max - t_min) * torch.rand(total_events, device=device, generator=self.rng)
                    # Add small time clustering for burst effect
                    time_jitter = torch.randn(total_events, device=device, generator=self.rng) * 0.02
                    times = (base_times + time_jitter).clamp(t_min, t_max)
                    
                    # Repeat coordinates according to events per pixel
                    hot_rep = active_hot_pixels.repeat_interleave(events_per_pixel, dim=0)
                    pol = self.hot_pol[active_indices].repeat_interleave(events_per_pixel) if c.polarity_consistent else self._pol(total_events, random=True)
                    
                    hot_events = torch.stack([times,
                                            hot_rep[:,0].float(),
                                            hot_rep[:,1].float(),
                                            pol], dim=1)
                    # Expand to batch size [B, N, 4]
                    hot_events = hot_events.unsqueeze(0).expand(batch_size, -1, -1)
                    events.append(hot_events)

            # Flicker sources (emit only at high sine) - more realistic street light flicker
            if c.flicker_sources>0:
                # Use midpoint time for phase calculation
                time_scalar = (t_min + t_max) / 2.0
                phase = self.flicker_phase + 2 * math.pi * c.flicker_freq_hz * time_scalar  # [flicker_sources]
                intensity = (torch.sin(phase) + 1.0) / 2.0  # [0,1]
                # Random per-source activation threshold in [0.7,0.9]
                flicker_threshold = 0.7 + torch.rand(c.flicker_sources, device=device, generator=self.rng) * 0.2
                active = intensity > flicker_threshold  # boolean mask [flicker_sources]
                if active.any():
                    centers = self.flicker_centers[active]  # [K,2]
                    K = centers.shape[0]
                    per = max(1, c.flicker_events // max(1, K))
                    n_events = K * per
                    # Angles & radii define offsets (ensure 1D shapes)
                    angles = torch.rand(n_events, device=device, generator=self.rng) * 2 * math.pi  # [n_events]
                    radii = torch.sqrt(torch.rand(n_events, device=device, generator=self.rng)) * 10.0  # [n_events]
                    offs = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)  # [n_events,2]
                    base = centers.repeat_interleave(per, dim=0) + offs  # [n_events,2]
                    base[:,0].clamp_(0, self.width-1)
                    base[:,1].clamp_(0, self.height-1)
                    times = t_min + (t_max - t_min) * torch.rand(n_events, device=device, generator=self.rng)  # [n_events]
                    pol = self._pol(n_events, random=True)  # [n_events]
                    # Assertions & shape validation
                    assert times.ndim == 1 and base.ndim == 2 and pol.ndim == 1, (
                        f"Dim mismatch (times:{times.shape}, base:{base.shape}, pol:{pol.shape})")
                    assert base.shape[0] == times.shape[0] == pol.shape[0], (
                        f"Length mismatch (times:{times.shape[0]}, base:{base.shape[0]}, pol:{pol.shape[0]})")
                    try:
                        flicker_events = torch.stack([times, base[:,0], base[:,1], pol], dim=1)  # [n_events,4]
                    except RuntimeError as e:
                        raise RuntimeError(
                            f"Flicker stacking failed: times {times.shape}, base {base.shape}, pol {pol.shape}. Error: {e}") from e
                    # Expand to batch size [B, N, 4]
                    flicker_events = flicker_events.unsqueeze(0).expand(batch_size, -1, -1)
                    events.append(flicker_events)

            # Lamp-like blobs (stationary with slow drift and intensity variation)
            if c.lamp_blobs > 0:
                dt = t_max - t_min
                # Very slow drift update
                self.lamp_pos += self.lamp_vel * (dt * self.height * 0.1)  # Much slower than moving blobs
                
                # Keep lamps in bounds with soft bouncing
                for i in range(c.lamp_blobs):
                    if self.lamp_pos[i, 0] <= 0 or self.lamp_pos[i, 0] >= self.width-1:
                        self.lamp_vel[i, 0] *= -0.8  # Damped bounce
                        self.lamp_pos[i, 0] = torch.clamp(self.lamp_pos[i, 0], 0, self.width-1)
                    if self.lamp_pos[i, 1] <= 0 or self.lamp_pos[i, 1] >= self.height-1:
                        self.lamp_vel[i, 1] *= -0.8  # Damped bounce  
                        self.lamp_pos[i, 1] = torch.clamp(self.lamp_pos[i, 1], 0, self.height-1)
                
                for i in range(c.lamp_blobs):
                    # Calculate intensity variation for this lamp
                    time_mid = (t_min + t_max) / 2
                    intensity_phase = self.lamp_intensity_phase[i] + time_mid * 0.5  # Slow variation
                    intensity_factor = 0.7 + 0.3 * torch.sin(intensity_phase)  # 0.7 to 1.0 intensity
                    
                    n = int(c.lamp_events * intensity_factor // c.lamp_blobs)
                    if n == 0: continue
                    
                    # Generate events in realistic lamp illumination pattern
                    # Core bright region
                    core_events = n // 3
                    if core_events > 0:
                        core_offsets = torch.randn(core_events, 2, device=device, generator=self.rng) * (c.lamp_radius * 0.3)
                        core_coords = self.lamp_pos[i].unsqueeze(0) + core_offsets
                        core_coords[:,0].clamp_(0, self.width-1)
                        core_coords[:,1].clamp_(0, self.height-1)
                        core_times = t_min + (t_max - t_min) * torch.rand(core_events, device=device, generator=self.rng)
                        core_pol = torch.ones(core_events, device=device)  # Positive polarity for bright core
                        core_lamp_events = torch.stack([core_times, core_coords[:,0], core_coords[:,1], core_pol], dim=1)
                        events.append(core_lamp_events.unsqueeze(0).expand(batch_size, -1, -1))
                    
                    # Medium illumination ring
                    medium_events = n // 2
                    if medium_events > 0:
                        # Ring pattern with more spread
                        angles = torch.rand(medium_events, device=device, generator=self.rng) * 2 * math.pi
                        radii = (torch.rand(medium_events, device=device, generator=self.rng) * 0.5 + 0.3) * c.lamp_radius
                        medium_offsets = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)
                        medium_coords = self.lamp_pos[i].unsqueeze(0) + medium_offsets
                        medium_coords[:,0].clamp_(0, self.width-1)
                        medium_coords[:,1].clamp_(0, self.height-1)
                        medium_times = t_min + (t_max - t_min) * torch.rand(medium_events, device=device, generator=self.rng)
                        medium_pol = self._pol(medium_events, random=True)  # Mixed polarity
                        medium_lamp_events = torch.stack([medium_times, medium_coords[:,0], medium_coords[:,1], medium_pol], dim=1)
                        events.append(medium_lamp_events.unsqueeze(0).expand(batch_size, -1, -1))
                    
                    # Outer dim glow
                    outer_events = n - core_events - medium_events
                    if outer_events > 0:
                        outer_offsets = torch.randn(outer_events, 2, device=device, generator=self.rng) * c.lamp_radius
                        outer_coords = self.lamp_pos[i].unsqueeze(0) + outer_offsets
                        outer_coords[:,0].clamp_(0, self.width-1)
                        outer_coords[:,1].clamp_(0, self.height-1)
                        outer_times = t_min + (t_max - t_min) * torch.rand(outer_events, device=device, generator=self.rng)
                        outer_pol = self._pol(outer_events, random=True)
                        outer_lamp_events = torch.stack([outer_times, outer_coords[:,0], outer_coords[:,1], outer_pol], dim=1)
                        events.append(outer_lamp_events.unsqueeze(0).expand(batch_size, -1, -1))

            # Moving blobs (vehicles, etc. - faster movement)
            if c.moving_blobs>0:
                # Update positions
                dt = t_max - t_min
                self.blob_pos += self.blob_vel * (dt * self.height)  # scale velocity
                self.blob_pos[:,0] = self.blob_pos[:,0] % self.width
                self.blob_pos[:,1] = self.blob_pos[:,1] % self.height
                for i in range(c.moving_blobs):
                    n = c.blob_events // c.moving_blobs
                    if n==0: continue
                    # Create trail-like pattern for moving objects
                    offsets = torch.randn(n,2, device=device, generator=self.rng)*c.blob_radius
                    # Add some motion blur effect
                    motion_blur = self.blob_vel[i].unsqueeze(0) * torch.linspace(-0.5, 0.5, n, device=device).unsqueeze(1)
                    coords = self.blob_pos[i].unsqueeze(0) + offsets + motion_blur
                    coords[:,0].clamp_(0,self.width-1)
                    coords[:,1].clamp_(0,self.height-1)
                    times = t_min + (t_max - t_min) * torch.rand(n, device=device, generator=self.rng)
                    pol = self._pol(n, random=True)
                    blob_events = torch.stack([times, coords[:,0], coords[:,1], pol], dim=1)
                    # Expand to batch size [B, N, 4]
                    blob_events = blob_events.unsqueeze(0).expand(batch_size, -1, -1)
                    events.append(blob_events)

            # Drifting edges (building edges, shadows)
            if c.drifting_edges>0 and c.edge_events > 0:
                for i in range(c.drifting_edges):
                    time_mid = (t_min + t_max) / 2
                    pos = (self.edge_pos0[i] + self.edge_dir[i]*time_mid*0.4) % 1.0  # Slower drift
                    n = c.edge_events // c.drifting_edges
                    if n==0: continue
                    times = t_min + (t_max - t_min) * torch.rand(n, device=device, generator=self.rng)
                    
                    if self.edge_orient[i]==0:  # Horizontal edge
                        y = pos*(self.height-1)
                        xs = torch.randint(0,self.width,(n,), generator=self.rng, device=device).float()
                        # Create more coherent edge with some thickness
                        edge_thickness = 3
                        ys = (y + torch.randn(n, device=device, generator=self.rng)*edge_thickness).clamp(0,self.height-1)
                    else:  # Vertical edge
                        x = pos*(self.width-1)
                        ys = torch.randint(0,self.height,(n,), generator=self.rng, device=device).float()
                        edge_thickness = 3
                        xs = (x + torch.randn(n, device=device, generator=self.rng)*edge_thickness).clamp(0,self.width-1)
                    
                    # Create alternating polarity pattern for edge
                    pol = torch.where(torch.arange(n, device=device) % 2 == 0, 1.0, -1.0)
                    edge_events = torch.stack([times,xs,ys,pol], dim=1)
                    # Expand to batch size [B, N, 4]
                    edge_events = edge_events.unsqueeze(0).expand(batch_size, -1, -1)
                    events.append(edge_events)

            # Grid interference (electrical interference, sensor artifacts)
            if c.grid_enabled and self.grid_x is not None and c.grid_events > 0:
                n = c.grid_events
                # Create periodic interference pattern
                time_mid = (t_min + t_max) / 2
                interference_phase = time_mid * 30.0  # 30 Hz interference
                if torch.sin(torch.tensor(interference_phase * 2 * math.pi)).item() > 0.3:  # Periodic activation
                    gx = self.grid_x[torch.randint(0,len(self.grid_x),(n,), generator=self.rng, device=device)]
                    gy = self.grid_y[torch.randint(0,len(self.grid_y),(n,), generator=self.rng, device=device)]
                    times = t_min + (t_max - t_min) * torch.rand(n, device=device, generator=self.rng)
                    # Create alternating polarity grid pattern
                    grid_pattern = ((gx // c.grid_spacing) + (gy // c.grid_spacing)) % 2
                    pol = torch.where(grid_pattern == 0, 1.0, -1.0)
                    grid_events = torch.stack([times,gx.float(),gy.float(),pol], dim=1)
                    # Expand to batch size [B, N, 4]
                    grid_events = grid_events.unsqueeze(0).expand(batch_size, -1, -1)
                    events.append(grid_events)

            if not events:
                return None

            out = torch.cat(events, dim=1)  # Concatenate along event dimension
            # Cap per batch
            if out.shape[1] > c.max_total_events:
                # Sample random events per batch
                for b in range(batch_size):
                    idx = torch.randperm(out.shape[1], generator=self.rng, device=device)[:c.max_total_events]
                    if b == 0:
                        final_out = out[b:b+1, idx]
                    else:
                        final_out = torch.cat([final_out, out[b:b+1, idx]], dim=0)
                out = final_out
            
            # Apply persistent augmentations to noise events
            out = self.apply_persistent_augmentations(out)
            # Shape [B,M,4]
        return out
    def _init_persistent_augmentations(self):
        """Initialize persistent augmentation parameters that stay constant for the entire video sequence"""
        if not self.cfg.enable_augmentations or not self.training:
            self.augmentation_state = {
                'enabled': False
            }
            return
        # Decide if augmentations are enabled for this sequence
        aug_enabled = torch.rand(1, generator=self.rng, device=self.device).item() < self.cfg.aug_prob
        
        if not aug_enabled:
            self.augmentation_state = {
                'enabled': False
            }
            return
            
        c = self.cfg
        self.augmentation_state = {
            'enabled': True,
            # Temporal jitter
            'temporal_jitter_enabled': torch.rand(1, generator=self.rng, device=self.device).item() < c.temporal_jitter_prob,
            'temporal_jitter_std': c.temporal_jitter_std,
            
            # Spatial jitter  
            'spatial_jitter_enabled': torch.rand(1, generator=self.rng, device=self.device).item() < c.spatial_jitter_prob,
            'spatial_jitter_std': c.spatial_jitter_std,
            
            # Dropout
            'dropout_enabled': torch.rand(1, generator=self.rng, device=self.device).item() < c.dropout_prob,
            'dropout_rate': torch.rand(1, generator=self.rng, device=self.device).item() * (c.dropout_rate_range[1] - c.dropout_rate_range[0]) + c.dropout_rate_range[0],
            'dropout_is_cluster': torch.rand(1, generator=self.rng, device=self.device).item() < c.cluster_dropout_prob,
            
            # Polarity flip
            'polarity_flip_enabled': torch.rand(1, generator=self.rng, device=self.device).item() < c.polarity_flip_prob,
            'polarity_flip_rate': torch.rand(1, generator=self.rng, device=self.device).item() * (c.polarity_flip_rate_range[1] - c.polarity_flip_rate_range[0]) + c.polarity_flip_rate_range[0],
            
            # Rate variation
            'rate_variation_enabled': torch.rand(1, generator=self.rng, device=self.device).item() < c.rate_variation_prob,
            'rate_factor': torch.rand(1, generator=self.rng, device=self.device).item() * (c.rate_factor_range[1] - c.rate_factor_range[0]) + c.rate_factor_range[0],
        }
        # Initialize persistent spatial dropout clusters if cluster dropout is enabled
        if self.augmentation_state['dropout_enabled'] and self.augmentation_state['dropout_is_cluster']:
            n_clusters = torch.randint(1, 5, (1,), generator=self.rng, device=self.device).item()
            self.augmentation_state['dropout_clusters'] = []
            for _ in range(n_clusters):
                center_x = torch.rand(1, generator=self.rng, device=self.device).item() * self.width
                center_y = torch.rand(1, generator=self.rng, device=self.device).item() * self.height
                radius = (torch.rand(1, generator=self.rng, device=self.device).item() * 0.2 + 0.1) * min(self.width, self.height)
                self.augmentation_state['dropout_clusters'].append({
                    'center_x': center_x,
                    'center_y': center_y,
                    'radius': radius
                })

    def apply_persistent_augmentations(self, events):
        """Apply persistent augmentations that were initialized once and stay constant throughout the video"""
        if not self.augmentation_state.get('enabled', False):
            return events
            
        B, N, _ = events.shape
        device = events.device
        augmented = events.clone()
        
        # 1. Temporal jitter (persistent noise characteristics)
        if self.augmentation_state['temporal_jitter_enabled']:
            time_jitter = torch.randn_like(augmented[:, :, 0]) * self.augmentation_state['temporal_jitter_std']
            augmented[:, :, 0] = (augmented[:, :, 0] + time_jitter).clamp(0, 1)
        
        # 2. Spatial jitter (persistent sensor misalignment)
        if self.augmentation_state['spatial_jitter_enabled']:
            spatial_jitter_x = torch.randn_like(augmented[:, :, 1]) * self.augmentation_state['spatial_jitter_std']
            spatial_jitter_y = torch.randn_like(augmented[:, :, 2]) * self.augmentation_state['spatial_jitter_std']
            augmented[:, :, 1] = (augmented[:, :, 1] + spatial_jitter_x).clamp(0, self.width - 1)
            augmented[:, :, 2] = (augmented[:, :, 2] + spatial_jitter_y).clamp(0, self.height - 1)
        
        # 3. Event dropout (persistent dead pixel regions or lighting conditions)
        if self.augmentation_state['dropout_enabled']:
            dropout_rate = self.augmentation_state['dropout_rate']
            
            if self.augmentation_state['dropout_is_cluster']:
                # Persistent spatial cluster dropout (dead pixel regions)
                keep_mask = torch.ones(B, N, dtype=torch.bool, device=device)
                for b in range(B):
                    for cluster in self.augmentation_state['dropout_clusters']:
                        center_x = cluster['center_x']
                        center_y = cluster['center_y'] 
                        radius = cluster['radius']
                        
                        dist_x = (augmented[b, :, 1] - center_x).abs()
                        dist_y = (augmented[b, :, 2] - center_y).abs()
                        cluster_mask = (dist_x < radius) & (dist_y < radius)
                        keep_mask[b] = keep_mask[b] & ~cluster_mask
            else:
                # Random dropout with persistent rate
                keep_mask = torch.rand(B, N, device=device, generator=self.rng) > dropout_rate
            
            # Apply dropout
            new_augmented = []
            for b in range(B):
                batch_events = augmented[b][keep_mask[b]]
                new_augmented.append(batch_events)
            
            if new_augmented:  # Check if any events remain
                min_events = min(batch.shape[0] for batch in new_augmented)
                
                final_augmented = torch.zeros(B, min(N, min_events), 4, device=device)
                for b in range(B):
                    n_events = min(min_events, new_augmented[b].shape[0])
                    if n_events > 0:
                        final_augmented[b, :n_events] = new_augmented[b][:n_events]
                
                # Pad back to original size if needed
                if final_augmented.shape[1] < N:
                    padding = torch.zeros(B, N - final_augmented.shape[1], 4, device=device)
                    augmented = torch.cat([final_augmented, padding], dim=1)
                else:
                    augmented = final_augmented
        
        # 4. Polarity flip (persistent sensor noise characteristics)
        if self.augmentation_state['polarity_flip_enabled']:
            flip_rate = self.augmentation_state['polarity_flip_rate']
            flip_mask = torch.rand(B, N, device=device, generator=self.rng) < flip_rate
            augmented[:, :, 3] = torch.where(flip_mask, -augmented[:, :, 3], augmented[:, :, 3])
        
        # 5. Event rate variation (persistent lighting/exposure conditions)
        if self.augmentation_state['rate_variation_enabled']:
            rate_factor = self.augmentation_state['rate_factor']
            if rate_factor < 1.0:  # Reduce events (low light conditions)
                keep_ratio = rate_factor
                keep_mask = torch.rand(B, N, device=device, generator=self.rng) < keep_ratio
                
                new_augmented = []
                for b in range(B):
                    batch_events = augmented[b][keep_mask[b]]
                    new_augmented.append(batch_events)
                
                if new_augmented:  # Check if any events remain
                    min_events = min(batch.shape[0] for batch in new_augmented)
                    
                    final_augmented = torch.zeros(B, min(N, min_events), 4, device=device)
                    for b in range(B):
                        n_events = min(min_events, new_augmented[b].shape[0])
                        if n_events > 0:
                            final_augmented[b, :n_events] = new_augmented[b][:n_events]
                    
                    # Pad back to original size if needed
                    if final_augmented.shape[1] < N:
                        padding = torch.zeros(B, N - final_augmented.shape[1], 4, device=device)
                        augmented = torch.cat([final_augmented, padding], dim=1)
                    else:
                        augmented = final_augmented
        
        return augmented
    
def merge_events_with_noise(original, noise):
    """
    Merge original events with persistent noise events
    
    Args:
        original: [B, N, 4] - original events
        noise: [B, M, 4] - noise events from PersistentNoiseGenerator
    
    Returns:
        merged: [B, N+M, 4] - merged events
    """
    if noise is None:
        return original
    
    # Ensure same device
    if noise.device != original.device:
        noise = noise.to(original.device)
    
    # Concatenate along the event dimension
    merged = torch.cat([original, noise], dim=1)
    
    return merged

def create_nighttime_noise_config():
    """
    Create a configuration optimized for night-time camera artifacts with strong illumination.
    This simulates real-world conditions where street lights, building lights, and camera
    sensor artifacts create persistent noise patterns that can cause mispredictions.
    
    Returns:
        PersistentNoiseConfig: Configuration optimized for night-time scenarios
    """
    return PersistentNoiseConfig(
        hot_pixels=800,              # More hot pixels for night conditions
        hot_pixel_rate=8,            # Higher rate for more visible artifacts
        flicker_sources=6,           # More street lights
        flicker_freq_hz=50.0,        # European power grid frequency
        flicker_events=500,          # More visible flicker
        lamp_blobs=4,                # Multiple street lamps/building lights
        lamp_radius=20,              # Larger illumination area
        lamp_events=1200,            # High intensity for bright lamps
        lamp_drift_speed=0.3,        # Very slow swaying
        lamp_intensity_var=0.4,      # More intensity variation
        moving_blobs=2,              # Some moving lights (vehicles)
        blob_radius=10,
        blob_events=400,
        blob_speed=3.0,              # Slower moving objects at night
        drifting_edges=2,            # Building edges, shadows
        edge_events=200,             # Visible edge artifacts
        grid_enabled=True,
        grid_spacing=24,             # Finer grid for more interference
        grid_events=100,             # Noticeable grid interference
        polarity_consistent=False,   # Mixed polarity for realism
        max_total_events=4000,       # Allow more events for rich night scenes
        prob_enable=0.9,             # High probability for night scenarios
        # Enhanced augmentation settings for realistic night conditions
        enable_augmentations=True,
        aug_prob=0.8,                # High probability for night conditions
        temporal_jitter_prob=0.6,
        temporal_jitter_std=0.15,    # More temporal noise at night
        spatial_jitter_prob=0.5,
        spatial_jitter_std=12.0,     # More spatial noise
        dropout_prob=0.6,
        dropout_rate_range=(0.15, 0.4),  # Higher dropout range
        cluster_dropout_prob=0.7,    # More cluster dropout (dead regions)
        polarity_flip_prob=0.3,
        polarity_flip_rate_range=(0.1, 0.25),
        rate_variation_prob=0.4,
        rate_factor_range=(0.4, 1.2)  # More rate variation
    )

def create_minimal_noise_config():
    """
    Create a minimal noise configuration for subtle augmentation.
    
    Returns:
        PersistentNoiseConfig: Minimal noise configuration
    """
    return PersistentNoiseConfig(
        hot_pixels=200,
        hot_pixel_rate=3,
        flicker_sources=1,
        flicker_events=100,
        lamp_blobs=1,
        lamp_events=300,
        moving_blobs=0,
        blob_events=0,
        drifting_edges=0,
        edge_events=0,
        grid_enabled=False,
        max_total_events=1000,
        prob_enable=0.5,
        # Minimal augmentation settings
        enable_augmentations=True,
        aug_prob=0.3,                # Lower probability for minimal setting
        temporal_jitter_prob=0.2,
        temporal_jitter_std=0.05,    # Minimal temporal noise
        spatial_jitter_prob=0.2,
        spatial_jitter_std=5.0,      # Minimal spatial noise
        dropout_prob=0.2,
        dropout_rate_range=(0.05, 0.15),  # Minimal dropout
        cluster_dropout_prob=0.3,
        polarity_flip_prob=0.1,
        polarity_flip_rate_range=(0.05, 0.15),
        rate_variation_prob=0.1,
        rate_factor_range=(0.7, 1.3)
    )

def create_persistent_noise_generator_with_augmentations(width, height, device, config_type='nighttime', training=True, seed=None):
    """
    Create a PersistentNoiseGenerator with appropriate configuration for persistent augmentations.
    
    Args:
        width, height: image dimensions
        device: torch device
        config_type: 'nighttime' or 'minimal' for different noise levels
        training: whether in training mode (enables augmentations)
        seed: random seed for reproducibility
    
    Returns:
        PersistentNoiseGenerator instance configured for persistent augmentations
    """
    if config_type == 'nighttime':
        config = create_nighttime_noise_config()
    elif config_type == 'minimal':
        config = create_minimal_noise_config()
    else:
        config = PersistentNoiseConfig()  # Default config
    
    return PersistentNoiseGenerator(width, height, device, config=config, seed=seed, training=training)