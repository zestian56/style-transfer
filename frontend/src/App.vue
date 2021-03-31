<template>
  <div class="max-w-full flex-col flex max-h-screen min-h-screen ">
    <div
      class="container flex items-center justify-center max-w-full flex-grow  overflow-y-hidden flex-wrap"
    >
      <WelcomeCard v-if="step === 0" :onClick="handleNextStep" />
      <PickImage
        v-if="step === 1 || displayAll"
        v-model="contentImage"
        id="contentImage "
        title="Content image"
        @load="handleLoadImage('contentImage')"
      />
      <PickImage
        v-if="step === 1 && displayStyleImage"
        v-model="styleImage"
        id="styleImage"
        title="Style image"
        @load="handleLoadImage('styleImage')"
      />
      <ModelConfig
        v-if="step === 1 && displayConfig"
        @onSubmit="handleSubmit"
      />
      <Loading
        v-if="step === 2"
        :progress="progress"
        :progressState="progressState"
        :image="resultImage"
      />
    </div>
    <button  @click="restartProcess" class="mt-4 btn btn-orange flex-shrink-0 shadow-md restart-btn" v-if="finished || (step !== 0 && progress === 0)">
      Restart process
    </button>
  </div>
</template>

<script>
import io from "socket.io-client";
import WelcomeCard from "./components/WelcomeCard.vue";
import PickImage from "./components/PickImage.vue";
import ModelConfig from "./components/ModelConfig.vue";
import Loading from "./components/Loading.vue";
import { arrayBufferToBase64 } from "./helpers";

export default {
  name: "App",
  components: { WelcomeCard, PickImage, ModelConfig, Loading },
  created() {
    this.socket = io.connect("http://localhost:5000");
    this.socket.on("connect", () => {
      console.log("Connect", this.contentImage);
    });
    this.socket.on("updateProcess", (data) => {
      this.progress = data.progress;
      this.progressState = data.state;
      if (data.img) {
        this.resultImage =
          "data:image/png;base64," + arrayBufferToBase64(data.img);
      }
    });
    this.socket.on("endProcess", () => {
      this.finished = true;
    });
  },
  data: function () {
    return {
      step: 0,
      contentImage: "",
      styleImage: "",
      resultImage: "",
      displayAll: false,
      displayStyleImage: false,
      displayConfig: false,
      progress: 0,
      modelConfig: {},
      progressState: "",
      finished: false,
    };
  },
  methods: {
    restartProcess: function() {
      this.step = 0;
      this.contentImage = "";
      this.styleImage = "";
      this.resultImage = "";
      this.progress = 0;
      this.finished = false;
      this.displayConfig = false;
      this.displayStyleImage = false;
      this.displayAll = false;
    },
    handleNextStep: function () {
      if (this.step <= 5) {
        this.step = this.step + 1;
      }
    },
    handleLoadImage: function (name) {
      if (name === "contentImage") {
        this.displayStyleImage = true;
      }
      if (name === "styleImage") {
        this.displayConfig = true;
      }
    },
    handleSubmit: function (values) {
      this.modelConfig = values;
      this.handleStartProcess();
    },
    handleStartProcess: function () {
      this.step = 2;
      this.resultImage = this.contentImage;
      const body = {
        content: this.contentImage,
        style: this.styleImage,
        show_every: +this.modelConfig.showEvery,
        steps: +this.modelConfig.steps,
      };
      this.socket.emit("startProcess", body);
    },
  },
};
</script>

<style scoped>
.container {
  z-index: 2;
}
.container:before {
  content: "";
  position: absolute;
  height: 2000px;
  width: 2000px;
  top: -10%;
  right: 48%;
  transform: translateY(-50%);
  background-image: linear-gradient(-45deg, #4481eb 0%, #04befe 100%);
  transition: 1.8s ease-in-out;
  border-radius: 50%;
  z-index: -1;
}

.restart-btn {
  z-index: 2;
}
</style>
