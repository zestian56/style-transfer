<template>
  <div class="m-6 max-w-md bg-white shadow-md rounded-xl p-9 w-full test">
    <div class="text-xl mb-4 font-medium text-black">{{ title }}</div>
    <div class="input-field">
      <i class="fas fa-image"></i>
      <input
        type="text"
        placeholder="Content Image URL"
        :id="id"
        @input="handleInput"
        :value="modelValue"
      />
    </div>
    <div class="img-container rounded-xl">
      <div>
        <p v-if="!modelValue || error" class="text-gray-500 text-center">
          <i class="far fa-file-image text-5xl mb-2"></i>
          <br />
          <span v-if="error"> Please write a valid image url. </span>
        </p>
      </div>
      <img
        :src="modelValue"
        v-if="modelValue"
        @error="handleError"
        @load="handleCompleted"
      />
    </div>
  </div>
</template>

<script>
export default {
  props: ["modelValue", "id", "title"],
  data: function () {
    return {
      error: false,
    };
  },
  methods: {
    handleInput(e) {
      this.$emit("update:modelValue", e.target.value);
    },
    handleBlur(e) {
      this.$emit("blur", e);
    },
    handleError(e) {
      this.error = true;
      this.$emit("error", e);
    },
    handleCompleted(e) {
      this.error = false;
      this.$emit("load", e);
    },
  },
};
</script>

<style scoped>
.img-container {
  height: 360px;
  width: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: rgba(0, 0, 0, 0.1);
}
.img-container img {
  width: 100%;
}

.test {
  min-height: 50px;
  transition: all 0.3s cubic-bezier(0.075, 0.82, 0.165, 1);
}
</style>