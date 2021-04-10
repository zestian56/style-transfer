import { shallowMount } from "@vue/test-utils";
import PickImage from "@/components/PickImage.vue";

describe("PickImage.vue", () => {
  it("renders props.msg when passed", () => {
    const msg = "new message";
    const wrapper = shallowMount(PickImage, {
      props: { msg },
    });
    expect(wrapper).toMatchSnapshot();
  });
});
