<template>
  <div class="root rounded-xl">
    <div class="progress rounded-r-xl" :style="{ width: progress + '%' }"></div>
  </div>
</template>

<script>
export default {
  props: ["progress"],
};
</script>

<style scoped>
.root {
  width: 100%;
  height: 25px;
  border: solid 1px rgba(0, 0, 0, 0.3);
  background-color: rgba(255, 255, 255, 0.9);
  overflow: hidden;
}
.progress {
  background-color: #ff9b45;
  height: 100%;
  transition: width .3s ease-in-out;
}
</style>