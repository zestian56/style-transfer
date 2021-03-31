<template>
  <form class="m-6 max-w-md flex items-stretch" @submit.prevent="handleSubmit">
    <div
      class="flex flex-col max-w-full p-6 py-9 bg-white shadow-md rounded-l-xl"
    >
      <div class="text-xl font-medium text-black">Model configuration</div>
      <div class="input-field">
        <i class="fas fa-walking"></i>
        <input type="number" placeholder="Steps" v-model="steps" min="1000" />
      </div>
      <div class="input-field">
        <i class="fas fa-search"></i>
        <input
          type="number"
          placeholder="Show every #"
          v-model="showEvery"
          min="5"
          max="50"
          required
        />
      </div>
      <button class="mt-4 btn btn-orange flex-shrink-0 shadow-md" >
        Start
      </button>
    </div>
  </form>
</template>

<script>
export default {
  data() {
    return {
      showEvery: undefined,
      steps: undefined,
    };
  },
  methods: {
    handleSubmit() {
      console.log("elpepe")
      this.$emit("onSubmit", {
        showEvery: this.showEvery,
        steps: this.steps,
      });
    },
  },
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