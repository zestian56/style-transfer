<template>
    <div class="m-6 max-w-md flex items-stretch">
      <div
        class="flex flex-col max-w-full p-6 py-9 bg-white shadow-md rounded-l-xl"
      >
        <div class="text-xl font-medium text-black">Style transfer</div>
        <p class="text-gray-500">Transform your photos into paintings!</p>
      </div>
      <button class="btn btn-orange flex-shrink-0 button-arrow shadow-md" @click="onClick">
        Start
      </button>
    </div>
</template>

<script>
export default {
    props: ["onClick"]
};
</script>

<style scoped>
.button-arrow {
  border-top-right-radius: 50%;
  border-bottom-right-radius: 50%;
  padding-left: 2rem;
  padding-right: 2rem;
  font-size: 1.3rem;
  letter-spacing: 2px;
}
.button-arrow:active,
.button-arrow:focus {
  outline: none;
}
.button-arrow:hover {
  transform: scale3d(2);
}
</style>