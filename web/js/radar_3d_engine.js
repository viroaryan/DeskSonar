/**
 * DeskSonar 3D Spatial Holographic Radar Engine (Three.js)
 * Next-Gen Spatial Computing Visualizer:
 * - 3D Translucent 20cm Origin Spherical Geofence with dynamic breathing aura
 * - 3D Real-Time Hand Bounding Box Dimensions (L x W x H in cm)
 * - 3D Cyberpunk Laptop with dynamic screen tilt & keyboard backlighting
 * - 3D Reflective Desk Surface with physical tap shockwave ripples
 * - 3D Real-Time Hand/Finger Tracking Avatar with Particle Trail
 */

class Radar3DEngine {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.width = this.container.clientWidth || 560;
    this.height = this.container.clientHeight || 420;

    // Three.js Core Components
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;

    // 3D Visual Objects
    this.laptopGroup = null;
    this.laptopScreenGroup = null;
    this.deskMesh = null;
    this.handAvatar = null;
    this.handBBoxMesh = null;
    this.geofenceSphere = null;
    this.geofenceAura = null;
    this.handTrail = [];
    this.maxTrailPoints = 40;
    this.trailLine = null;
    this.ultrasoundCone = null;
    this.ripples = [];

    // Animation state
    this.wavePhase = 0;
    this.currentHandPos = { x: 0, y: 0.12, z: 0.15 };
    this.targetHandPos = { x: 0, y: 0.12, z: 0.15 };
    this.currentBBoxSize = { l: 0.08, w: 0.08, h: 0.04 };
    this.isInGeofence = true;

    this.init();
  }

  init() {
    if (typeof THREE === 'undefined') {
      console.warn('Three.js not loaded. 3D Engine waiting for library...');
      return;
    }

    // 1. Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x020611);
    this.scene.fog = new THREE.FogExp2(0x020611, 0.5);

    // 2. Camera setup
    this.camera = new THREE.PerspectiveCamera(50, this.width / this.height, 0.01, 20);
    this.camera.position.set(0, 0.38, 0.52);
    this.camera.lookAt(0, 0.10, 0.15);

    // 3. Renderer setup
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.innerHTML = '';
    this.container.appendChild(this.renderer.domElement);

    // 4. OrbitControls
    if (typeof THREE.OrbitControls !== 'undefined') {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.maxPolarAngle = Math.PI / 2 - 0.02;
      this.controls.minDistance = 0.15;
      this.controls.maxDistance = 1.8;
    }

    // 5. Cinematic Lighting
    const ambientLight = new THREE.AmbientLight(0x00f0ff, 0.5);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(0.6, 1.2, 0.6);
    this.scene.add(dirLight);

    const cyanPointLight = new THREE.PointLight(0x00f0ff, 2.0, 1.5);
    cyanPointLight.position.set(0, 0.25, 0.05);
    this.scene.add(cyanPointLight);

    const greenRimLight = new THREE.PointLight(0x00ff88, 1.8, 1.2);
    greenRimLight.position.set(-0.3, 0.2, 0.25);
    this.scene.add(greenRimLight);

    // 6. Build 3D World Elements
    this._buildDeskSurface();
    this._buildLaptopModel();
    this._buildGeofenceSphere();
    this._buildHandAvatar();
    this._buildHandBoundingBox();
    this._buildUltrasoundBeam();
    this._buildGrid();

    // 7. Handle Resize
    window.addEventListener('resize', () => this.onWindowResize());

    // 8. Start Render Loop
    this.animate();
  }

  _buildGrid() {
    const gridHelper = new THREE.GridHelper(1.8, 36, 0x00f0ff, 0x07273e);
    gridHelper.position.y = -0.001;
    this.scene.add(gridHelper);
  }

  _buildDeskSurface() {
    const deskGeo = new THREE.BoxGeometry(1.8, 0.02, 1.2);
    const deskMat = new THREE.MeshStandardMaterial({
      color: 0x050c18,
      roughness: 0.2,
      metalness: 0.85
    });
    this.deskMesh = new THREE.Mesh(deskGeo, deskMat);
    this.deskMesh.position.set(0, -0.01, 0.35);
    this.scene.add(this.deskMesh);
  }

  _buildGeofenceSphere() {
    // 20cm radius spherical geofence from microphone origin (0, 0.15, 0.10)
    const sphereGeo = new THREE.SphereGeometry(0.20, 32, 32);
    const sphereMat = new THREE.MeshBasicMaterial({
      color: 0x00ff88,
      wireframe: true,
      transparent: true,
      opacity: 0.25
    });
    this.geofenceSphere = new THREE.Mesh(sphereGeo, sphereMat);
    this.geofenceSphere.position.set(0, 0.15, 0.10);
    this.scene.add(this.geofenceSphere);

    // Subtle inner translucent glow
    const auraGeo = new THREE.SphereGeometry(0.196, 24, 24);
    const auraMat = new THREE.MeshBasicMaterial({
      color: 0x00ff88,
      transparent: true,
      opacity: 0.04,
      side: THREE.BackSide
    });
    this.geofenceAura = new THREE.Mesh(auraGeo, auraMat);
    this.geofenceSphere.add(this.geofenceAura);
  }

  _buildLaptopModel() {
    this.laptopGroup = new THREE.Group();

    // Base (Keyboard deck)
    const baseGeo = new THREE.BoxGeometry(0.32, 0.012, 0.22);
    const baseMat = new THREE.MeshStandardMaterial({ color: 0x0b1320, metalness: 0.9, roughness: 0.25 });
    const base = new THREE.Mesh(baseGeo, baseMat);
    this.laptopGroup.add(base);

    // Glowing Cyan Keyboard
    const kbGeo = new THREE.PlaneGeometry(0.26, 0.11);
    const kbMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.5 });
    const kb = new THREE.Mesh(kbGeo, kbMat);
    kb.rotation.x = -Math.PI / 2;
    kb.position.set(0, 0.007, -0.02);
    this.laptopGroup.add(kb);

    // Stereo Speaker Emitters with Green Glow
    const spkGeo = new THREE.CylinderGeometry(0.012, 0.012, 0.005, 16);
    const spkMat = new THREE.MeshBasicMaterial({ color: 0x00ff88 });

    const spkLeft = new THREE.Mesh(spkGeo, spkMat);
    spkLeft.position.set(-0.13, 0.007, -0.02);
    this.laptopGroup.add(spkLeft);

    const spkRight = new THREE.Mesh(spkGeo, spkMat);
    spkRight.position.set(0.13, 0.007, -0.02);
    this.laptopGroup.add(spkRight);

    // Screen
    this.laptopScreenGroup = new THREE.Group();
    const screenGeo = new THREE.BoxGeometry(0.32, 0.22, 0.008);
    const screenMat = new THREE.MeshStandardMaterial({ color: 0x080f18, metalness: 0.85, roughness: 0.2 });
    const screenBack = new THREE.Mesh(screenGeo, screenMat);
    screenBack.position.set(0, 0.11, 0);
    this.laptopScreenGroup.add(screenBack);

    // Glowing Display Panel
    const dispGeo = new THREE.PlaneGeometry(0.30, 0.19);
    const dispMat = new THREE.MeshBasicMaterial({ color: 0x021020 });
    const disp = new THREE.Mesh(dispGeo, dispMat);
    disp.position.set(0, 0.11, 0.005);
    this.laptopScreenGroup.add(disp);

    // Dual MEMS Bezel Microphones (Intel SST)
    const micGeo = new THREE.SphereGeometry(0.004, 8, 8);
    const micMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
    const micL = new THREE.Mesh(micGeo, micMat);
    micL.position.set(-0.04, 0.21, 0.006);
    this.laptopScreenGroup.add(micL);

    const micR = new THREE.Mesh(micGeo, micMat);
    micR.position.set(0.04, 0.21, 0.006);
    this.laptopScreenGroup.add(micR);

    this.laptopScreenGroup.position.set(0, 0.006, -0.11);
    this.laptopScreenGroup.rotation.x = 0.26; // Default tilt (~105 deg)
    this.laptopGroup.add(this.laptopScreenGroup);

    this.laptopGroup.position.set(0, 0.006, 0);
    this.scene.add(this.laptopGroup);
  }

  _buildHandAvatar() {
    const handGroup = new THREE.Group();

    // Palm / Hand Core Sphere
    const coreGeo = new THREE.SphereGeometry(0.022, 24, 24);
    const coreMat = new THREE.MeshBasicMaterial({
      color: 0x00ff88,
      wireframe: true
    });
    const core = new THREE.Mesh(coreGeo, coreMat);
    handGroup.add(core);

    // Inner Glowing Core
    const innerGeo = new THREE.SphereGeometry(0.014, 16, 16);
    const innerMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.8
    });
    const inner = new THREE.Mesh(innerGeo, innerMat);
    handGroup.add(inner);

    // Concentric Holographic Ring
    const ringGeo = new THREE.RingGeometry(0.035, 0.042, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.6
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    handGroup.add(ring);
    this.handRing = ring;

    handGroup.position.set(0, 0.12, 0.15);
    this.handAvatar = handGroup;
    this.scene.add(this.handAvatar);

    // Trajectory Trail
    const trailGeo = new THREE.BufferGeometry();
    const positions = new Float32Array(this.maxTrailPoints * 3);
    trailGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const trailMat = new THREE.LineBasicMaterial({
      color: 0x00ff88,
      transparent: true,
      opacity: 0.5
    });
    this.trailLine = new THREE.Line(trailGeo, trailMat);
    this.scene.add(this.trailLine);
  }

  _buildHandBoundingBox() {
    // Dynamic Hand 3D Bounding Box (Length x Width x Height)
    const boxGeo = new THREE.BoxGeometry(0.08, 0.04, 0.08);
    const boxMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      wireframe: true,
      transparent: true,
      opacity: 0.4
    });
    this.handBBoxMesh = new THREE.Mesh(boxGeo, boxMat);
    this.handBBoxMesh.position.set(0, 0.12, 0.15);
    this.scene.add(this.handBBoxMesh);
  }

  _buildUltrasoundBeam() {
    const coneGeo = new THREE.ConeGeometry(0.18, 0.35, 24, 1, true);
    const coneMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      wireframe: true,
      transparent: true,
      opacity: 0.08,
      side: THREE.DoubleSide
    });
    this.ultrasoundCone = new THREE.Mesh(coneGeo, coneMat);
    this.ultrasoundCone.rotation.x = -Math.PI / 2 + 0.25;
    this.ultrasoundCone.position.set(0, 0.10, 0.18);
    this.scene.add(this.ultrasoundCone);
  }

  triggerDeskTapShockwave(x = 0, z = 0.25) {
    const rippleGeo = new THREE.RingGeometry(0.01, 0.03, 32);
    const rippleMat = new THREE.MeshBasicMaterial({
      color: 0xff0055,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9
    });
    const ripple = new THREE.Mesh(rippleGeo, rippleMat);
    ripple.rotation.x = -Math.PI / 2;
    ripple.position.set(x, 0.005, z);
    this.scene.add(ripple);
    this.ripples.push({ mesh: ripple, scale: 1.0, opacity: 0.9 });
  }

  updateAcousticTargets(spatial3D, targets, isLivingHuman = true, isTap = false, geometry = null, bbox = null) {
    if (spatial3D) {
      this.targetHandPos.x = THREE.MathUtils.clamp(spatial3D.x, -0.25, 0.25);
      this.targetHandPos.y = THREE.MathUtils.clamp(spatial3D.y, 0.02, 0.35);
      this.targetHandPos.z = THREE.MathUtils.clamp(spatial3D.z, 0.04, 0.40);
    }

    if (bbox) {
      this.currentBBoxSize.l = (bbox.length_cm || 10.0) * 0.01;
      this.currentBBoxSize.w = (bbox.width_cm || 8.0) * 0.01;
      this.currentBBoxSize.h = (bbox.height_cm || 4.0) * 0.01;
      this.isInGeofence = bbox.is_in_20cm_geofence;
    }

    if (geometry && geometry.screen_tilt_deg && this.laptopScreenGroup) {
      const tiltRad = (180.0 - geometry.screen_tilt_deg) * (Math.PI / 180.0);
      this.laptopScreenGroup.rotation.x = THREE.MathUtils.lerp(this.laptopScreenGroup.rotation.x, tiltRad, 0.1);
    }

    if (isTap) {
      this.triggerDeskTapShockwave(this.currentHandPos.x, this.currentHandPos.z);
    }
  }

  setCameraView(viewMode) {
    if (!this.controls) return;
    if (viewMode === 'perspective') {
      this.camera.position.set(0, 0.38, 0.52);
      this.controls.target.set(0, 0.10, 0.15);
    } else if (viewMode === 'top') {
      this.camera.position.set(0, 0.65, 0.18);
      this.controls.target.set(0, 0.05, 0.18);
    } else if (viewMode === 'side') {
      this.camera.position.set(0.55, 0.15, 0.18);
      this.controls.target.set(0, 0.10, 0.18);
    }
    this.controls.update();
  }

  onWindowResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    this.width = this.container.clientWidth;
    this.height = this.container.clientHeight;
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height);
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    this.wavePhase += 0.04;

    // Smooth Hand Position Lerp
    this.currentHandPos.x = THREE.MathUtils.lerp(this.currentHandPos.x, this.targetHandPos.x, 0.25);
    this.currentHandPos.y = THREE.MathUtils.lerp(this.currentHandPos.y, this.targetHandPos.y, 0.25);
    this.currentHandPos.z = THREE.MathUtils.lerp(this.currentHandPos.z, this.targetHandPos.z, 0.25);

    if (this.handAvatar) {
      this.handAvatar.position.set(this.currentHandPos.x, this.currentHandPos.y, this.currentHandPos.z);
      if (this.handRing) {
        this.handRing.rotation.z += 0.02;
        const scale = 1.0 + Math.sin(this.wavePhase * 2) * 0.08;
        this.handRing.scale.set(scale, scale, scale);
      }
    }

    // Dynamic Bounding Box Update
    if (this.handBBoxMesh) {
      this.handBBoxMesh.position.set(this.currentHandPos.x, this.currentHandPos.y, this.currentHandPos.z);
      this.handBBoxMesh.scale.set(
        Math.max(0.04, this.currentBBoxSize.w),
        Math.max(0.02, this.currentBBoxSize.h),
        Math.max(0.04, this.currentBBoxSize.l)
      );

      // Color change on geofence lock
      if (this.isInGeofence) {
        this.handBBoxMesh.material.color.setHex(0x00ff88);
      } else {
        this.handBBoxMesh.material.color.setHex(0xffb703);
      }
    }

    // Geofence Sphere Pulsing Aura
    if (this.geofenceSphere) {
      const pulse = 1.0 + Math.sin(this.wavePhase) * 0.015;
      this.geofenceSphere.scale.set(pulse, pulse, pulse);
      if (this.isInGeofence) {
        this.geofenceSphere.material.color.setHex(0x00ff88);
        this.geofenceSphere.material.opacity = 0.22 + Math.sin(this.wavePhase * 1.5) * 0.05;
      } else {
        this.geofenceSphere.material.color.setHex(0xff0055);
        this.geofenceSphere.material.opacity = 0.35;
      }
    }

    // Update Particle Trail
    this.handTrail.push(new THREE.Vector3(this.currentHandPos.x, this.currentHandPos.y, this.currentHandPos.z));
    if (this.handTrail.length > this.maxTrailPoints) {
      this.handTrail.shift();
    }
    if (this.trailLine) {
      const posAttr = this.trailLine.geometry.attributes.position;
      for (let i = 0; i < this.handTrail.length; i++) {
        posAttr.setXYZ(i, this.handTrail[i].x, this.handTrail[i].y, this.handTrail[i].z);
      }
      this.trailLine.geometry.setDrawRange(0, this.handTrail.length);
      posAttr.needsUpdate = true;
    }

    // Animate Tap Shockwave Ripples
    for (let i = this.ripples.length - 1; i >= 0; i--) {
      const r = this.ripples[i];
      r.scale += 0.08;
      r.opacity -= 0.025;
      r.mesh.scale.set(r.scale, r.scale, 1);
      r.mesh.material.opacity = Math.max(0, r.opacity);
      if (r.opacity <= 0) {
        this.scene.remove(r.mesh);
        this.ripples.splice(i, 1);
      }
    }

    if (this.controls) {
      this.controls.update();
    }

    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }
}
