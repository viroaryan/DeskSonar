/**
 * DeskSonar 3D Spatial Holographic Radar Engine (Three.js)
 * Visualizes:
 * - 3D Translucent 20cm Origin Spherical Geofence
 * - 3D Real-Time Hand Bounding Box Dimensions (L x W x H in cm)
 * - 3D Laptop with dynamic physical screen tilt angle
 * - 3D Physical Desk Surface with tap shockwave ripples
 * - 3D Real-Time Hand/Finger Tracking Avatar (X, Y, Z)
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
    this.handTrail = [];
    this.maxTrailPoints = 30;
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
    this.scene.background = new THREE.Color(0x04080e);
    this.scene.fog = new THREE.FogExp2(0x04080e, 0.6);

    // 2. Camera setup
    this.camera = new THREE.PerspectiveCamera(50, this.width / this.height, 0.01, 20);
    this.camera.position.set(0, 0.40, 0.55);
    this.camera.lookAt(0, 0.10, 0.15);

    // 3. Renderer setup
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.shadowMap.enabled = true;
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

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0x00f0ff, 0.4);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(0.5, 1.0, 0.5);
    this.scene.add(dirLight);

    const glowLight = new THREE.PointLight(0x00f0ff, 1.5, 1.2);
    glowLight.position.set(0, 0.2, 0);
    this.scene.add(glowLight);

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
    const gridHelper = new THREE.GridHelper(1.6, 32, 0x00f0ff, 0x0a2233);
    gridHelper.position.y = -0.001;
    this.scene.add(gridHelper);
  }

  _buildDeskSurface() {
    const deskGeo = new THREE.BoxGeometry(1.6, 0.02, 1.0);
    const deskMat = new THREE.MeshStandardMaterial({
      color: 0x08121f,
      roughness: 0.4,
      metalness: 0.8
    });
    this.deskMesh = new THREE.Mesh(deskGeo, deskMat);
    this.deskMesh.position.set(0, -0.01, 0.35);
    this.scene.add(this.deskMesh);
  }

  _buildGeofenceSphere() {
    // 20cm radius spherical geofence from microphone origin (0, 0.20, 0)
    const sphereGeo = new THREE.SphereGeometry(0.20, 24, 24);
    const sphereMat = new THREE.MeshBasicMaterial({
      color: 0x00ff88,
      wireframe: true,
      transparent: true,
      opacity: 0.20
    });
    this.geofenceSphere = new THREE.Mesh(sphereGeo, sphereMat);
    this.geofenceSphere.position.set(0, 0.15, 0.10);
    this.scene.add(this.geofenceSphere);
  }

  _buildLaptopModel() {
    this.laptopGroup = new THREE.Group();

    // Base (Keyboard deck)
    const baseGeo = new THREE.BoxGeometry(0.32, 0.012, 0.22);
    const baseMat = new THREE.MeshStandardMaterial({ color: 0x111923, metalness: 0.9, roughness: 0.2 });
    const base = new THREE.Mesh(baseGeo, baseMat);
    this.laptopGroup.add(base);

    // Glowing Keyboard
    const kbGeo = new THREE.PlaneGeometry(0.26, 0.11);
    const kbMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.4 });
    const kb = new THREE.Mesh(kbGeo, kbMat);
    kb.rotation.x = -Math.PI / 2;
    kb.position.set(0, 0.007, -0.02);
    this.laptopGroup.add(kb);

    // Stereo Speaker Emitters
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
    const screenMat = new THREE.MeshStandardMaterial({ color: 0x080f18, metalness: 0.8 });
    const screenBack = new THREE.Mesh(screenGeo, screenMat);
    screenBack.position.set(0, 0.11, 0);
    this.laptopScreenGroup.add(screenBack);

    // Glowing Display Panel
    const dispGeo = new THREE.PlaneGeometry(0.30, 0.19);
    const dispMat = new THREE.MeshBasicMaterial({ color: 0x041525 });
    const disp = new THREE.Mesh(dispGeo, dispMat);
    disp.position.set(0, 0.11, 0.005);
    this.laptopScreenGroup.add(disp);

    this.laptopScreenGroup.position.set(0, 0.006, -0.11);
    this.laptopScreenGroup.rotation.x = 0.26;
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
      transparent: true,
      opacity: 0.85
    });
    this.coreMesh = new THREE.Mesh(coreGeo, coreMat);
    handGroup.add(this.coreMesh);

    // Outer Holographic Aura
    const auraGeo = new THREE.SphereGeometry(0.035, 16, 16);
    const auraMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      wireframe: true,
      transparent: true,
      opacity: 0.4
    });
    this.auraMesh = new THREE.Mesh(auraGeo, auraMat);
    handGroup.add(this.auraMesh);

    // 5 Finger Nodes
    this.fingerNodes = [];
    for (let i = 0; i < 5; i++) {
      const fGeo = new THREE.SphereGeometry(0.005, 12, 12);
      const fMat = new THREE.MeshBasicMaterial({ color: 0x00ff88 });
      const fMesh = new THREE.Mesh(fGeo, fMat);
      const angle = (i - 2) * 0.35;
      fMesh.position.set(Math.sin(angle) * 0.03, 0.005, Math.cos(angle) * 0.03 + 0.012);
      handGroup.add(fMesh);
      this.fingerNodes.push(fMesh);
    }

    this.handAvatar = handGroup;
    this.handAvatar.position.set(0, 0.12, 0.15);
    this.scene.add(this.handAvatar);

    // 3D Motion Trail
    const maxPoints = this.maxTrailPoints;
    const positions = new Float32Array(maxPoints * 3);
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const trailMat = new THREE.LineBasicMaterial({
      color: 0x00ff88,
      transparent: true,
      opacity: 0.6,
      linewidth: 2
    });
    this.trailLine = new THREE.Line(trailGeo, trailMat);
    this.scene.add(this.trailLine);
  }

  _buildHandBoundingBox() {
    // 3D Wireframe Box showing live L x W x H
    const boxGeo = new THREE.BoxGeometry(0.08, 0.04, 0.08);
    const boxMat = new THREE.MeshBasicMaterial({
      color: 0x00ff88,
      wireframe: true,
      transparent: true,
      opacity: 0.75
    });
    this.handBBoxMesh = new THREE.Mesh(boxGeo, boxMat);
    this.handBBoxMesh.position.set(0, 0.12, 0.15);
    this.scene.add(this.handBBoxMesh);
  }

  _buildUltrasoundBeam() {
    const coneGeo = new THREE.ConeGeometry(0.35, 0.40, 24, 1, true);
    const coneMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      wireframe: true,
      transparent: true,
      opacity: 0.10,
      side: THREE.DoubleSide
    });
    this.ultrasoundCone = new THREE.Mesh(coneGeo, coneMat);
    this.ultrasoundCone.position.set(0, 0.02, 0.20);
    this.ultrasoundCone.rotation.x = Math.PI / 2;
    this.scene.add(this.ultrasoundCone);
  }

  triggerDeskTapRipple(x = 0, z = 0.15) {
    const rippleGeo = new THREE.RingGeometry(0.01, 0.02, 32);
    const rippleMat = new THREE.MeshBasicMaterial({
      color: 0xff0055,
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide
    });
    const ripple = new THREE.Mesh(rippleGeo, rippleMat);
    ripple.rotation.x = -Math.PI / 2;
    ripple.position.set(x, 0.002, z);
    this.scene.add(ripple);
    this.ripples.push({ mesh: ripple, radius: 0.02, opacity: 0.9 });
  }

  updateAcousticTargets(spatial3d, targets, isLiving, isTap, geometry, bbox) {
    if (geometry && this.laptopScreenGroup) {
      const tiltRad = (geometry.screen_tilt_deg - 90.0) * (Math.PI / 180.0);
      this.laptopScreenGroup.rotation.x = tiltRad;
    }

    if (bbox) {
      this.isInGeofence = bbox.is_in_20cm_geofence;
      this.currentBBoxSize.l = Math.max(0.04, (bbox.length_cm || 8.0) * 0.01);
      this.currentBBoxSize.w = Math.max(0.04, (bbox.width_cm || 8.0) * 0.01);
      this.currentBBoxSize.h = Math.max(0.02, (bbox.height_cm || 4.0) * 0.01);

      if (this.geofenceSphere) {
        this.geofenceSphere.material.color.setHex(this.isInGeofence ? 0x00ff88 : 0x556677);
      }
      if (this.handBBoxMesh) {
        this.handBBoxMesh.scale.set(
          this.currentBBoxSize.w / 0.08,
          this.currentBBoxSize.h / 0.04,
          this.currentBBoxSize.l / 0.08
        );
        this.handBBoxMesh.material.color.setHex(this.isInGeofence ? 0x00ff88 : 0x8899aa);
      }
    }

    if (!spatial3d) return;

    const rawX = spatial3d.x || 0.0;
    const rawY = Math.max(0.03, Math.min(0.35, spatial3d.y || 0.12));
    const rawZ = Math.max(0.04, Math.min(0.35, spatial3d.z || 0.15));

    this.targetHandPos.x = rawX;
    this.targetHandPos.y = rawY;
    this.targetHandPos.z = rawZ;

    if (isTap) {
      this.triggerDeskTapRipple(rawX, rawZ);
    }
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    if (this.controls) this.controls.update();

    // 1. Lerp Hand Avatar & Bounding Box Position
    const lerpFactor = 0.25;
    this.currentHandPos.x += (this.targetHandPos.x - this.currentHandPos.x) * lerpFactor;
    this.currentHandPos.y += (this.targetHandPos.y - this.currentHandPos.y) * lerpFactor;
    this.currentHandPos.z += (this.targetHandPos.z - this.currentHandPos.z) * lerpFactor;

    if (this.handAvatar) {
      this.handAvatar.position.set(
        this.currentHandPos.x,
        this.currentHandPos.y,
        this.currentHandPos.z
      );

      if (this.auraMesh) {
        this.auraMesh.rotation.y += 0.02;
        this.auraMesh.rotation.x += 0.01;
      }
    }

    if (this.handBBoxMesh) {
      this.handBBoxMesh.position.set(
        this.currentHandPos.x,
        this.currentHandPos.y,
        this.currentHandPos.z
      );
    }

    // 2. Update Motion Trail
    if (this.trailLine) {
      this.handTrail.push({ ...this.currentHandPos });
      if (this.handTrail.length > this.maxTrailPoints) {
        this.handTrail.shift();
      }

      const posAttr = this.trailLine.geometry.attributes.position;
      for (let i = 0; i < this.maxTrailPoints; i++) {
        const pt = this.handTrail[i] || this.currentHandPos;
        posAttr.setXYZ(i, pt.x, pt.y, pt.z);
      }
      posAttr.needsUpdate = true;
    }

    // 3. Pulse Acoustic Waves
    this.wavePhase += 0.04;
    if (this.ultrasoundCone) {
      const scale = 1.0 + 0.04 * Math.sin(this.wavePhase * 4);
      this.ultrasoundCone.scale.set(scale, 1.0, scale);
    }

    // 4. Animate Desk Tap Ripples
    for (let i = this.ripples.length - 1; i >= 0; i--) {
      const rip = this.ripples[i];
      rip.radius += 0.008;
      rip.opacity -= 0.025;
      rip.mesh.scale.set(rip.radius * 30, rip.radius * 30, 1);
      rip.mesh.material.opacity = Math.max(0, rip.opacity);

      if (rip.opacity <= 0) {
        this.scene.remove(rip.mesh);
        this.ripples.splice(i, 1);
      }
    }

    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  setCameraView(viewMode) {
    if (!this.camera) return;
    if (viewMode === 'top') {
      this.camera.position.set(0, 0.75, 0.20);
      this.camera.lookAt(0, 0, 0.20);
    } else if (viewMode === 'side') {
      this.camera.position.set(0.60, 0.20, 0.20);
      this.camera.lookAt(0, 0.1, 0.20);
    } else {
      this.camera.position.set(0, 0.40, 0.55);
      this.camera.lookAt(0, 0.10, 0.15);
    }
  }

  onWindowResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    this.width = this.container.clientWidth;
    this.height = this.container.clientHeight;
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height);
  }
}
