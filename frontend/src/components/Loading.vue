<template>
  <div class="root">
    <div class="pb-6 image-container">
      <img :src="image" />
    </div>
    <ProgressBarVue :progress="progress" />
    <div class="flex items-center justify-center pt-3 w-full text-gray-500">
      <p>{{ progressState }}</p>
    </div>
  </div>
</template>

<script>
import ProgressBarVue from "./ProgressBar.vue";

export default {
  props: ["progress", "steps", "progressState", "image"],
  components: { ProgressBarVue },
};
</script>

<style scoped>
.root {
  width: 100%;
  max-width: 500px;
}

.image-container{
  height: 360px;
}

.image-container img {
  max-height: 100%;
  margin: auto;
}
</style>